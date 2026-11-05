#!/usr/bin/env python3
"""
Branching Continuation Pilot
============================
回答问题：同题、同位置、相近 token 长度下，不同推理状态的 V 能否预测后续成功率？

三阶段：
  --mode traj  : 用 temperature>0 生成 N_TRAJ 条独立基础轨迹
  --mode cont  : 对每条轨迹截取 8/32/64 步前缀，每前缀用【统一预算】续跑 N_CONT 次
  --mode all   : 依次执行

关键控制（按评审要求）：
  1. 固定剩余预算：所有前缀都是 CONT_BUDGET 步，杜绝"位置-预算"混淆
  2. 记录 token position：prefix_ntok，供后续统计控制
  3. 记录终止状态：finished / success 分开存，避免 H3 混杂
  4. generate_step 会就地修改 messages，必须传副本

用法：
  CUDA_VISIBLE_DEVICES=0 python branch_pilot.py --mode all --shard 0 --nshards 2
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))

from eval_agent.qa_verifier import answers_equal, finalize_rollout, has_explicit_final_marker, split_steps
from eval_agent.qa_rollout_chunked_context import (
    generate_step, generate_iterative, make_conversations, get_question, get_gold,
    first_value,
)

ROLES = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
CONT_PROMPT = "Continue with exactly one concise new reasoning step."

MODEL = '/data3/xhy/codes/models/Qwen3-8B'
INPUT = str(ROOT / 'data/qa/olympiadbench_eval200_seed20261031.jsonl')


def log(msg):
    print('[%s] %s' % (time.strftime('%F %T'), msg), flush=True)


def load_model(path, device='cuda'):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={'': device} if device != 'auto' else 'auto')
    model.eval()
    return model, tok


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]


def append_jsonl(p, rec):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------- 阶段 1
def stage_traj(model, tok, rows, args, out_path):
    done = {(r['qid'], r['traj_idx']) for r in read_jsonl(out_path)}
    log('stage traj: 已完成 %d 条，开始补充' % len(done))
    for row in rows:
        qid = str(row.get('id', row.get('problem_idx', '')))
        question = get_question(row)
        gold = get_gold(row)
        if not gold:
            log('  [skip] %s gold 为空，跳过' % qid)
            continue
        for ti in range(args.n_traj):
            if (qid, ti) in done:
                continue
            t0 = time.time()
            raw = generate_iterative(
                model, tok, question, args.step_tokens, args.max_steps,
                args.temperature, args.context_max_tokens,
                wall_clock_timeout=args.wall_clock_timeout)
            normalized, fallback_pred, finished = finalize_rollout(raw, 'strict')
            steps = split_steps(normalized)
            if len(steps) > args.max_steps:
                steps = steps[:args.max_steps]
                finished = False
            if finished and not has_explicit_final_marker('\n'.join(steps)):
                finished = False
            pred = steps[-1] if steps else raw
            ok = bool(finished and (answers_equal(pred, gold) or answers_equal(raw, gold)))
            conv = make_conversations(question, steps)
            rec = {
                'qid': qid, 'traj_idx': ti, 'question': question, 'gold_answer': gold,
                'n_steps': len(steps), 'finished': bool(finished), 'success': ok,
                'pred_answer': pred, 'conversations': conv,
                'secs': round(time.time() - t0, 1),
            }
            append_jsonl(out_path, rec)
            log('  traj %s#%d steps=%d finished=%s ok=%s (%.1fs)' % (
                qid, ti, len(steps), finished, ok, time.time() - t0))


# ---------------------------------------------------------------- 阶段 2
def prefix_messages(conv, prefix_steps):
    """取前 2*prefix_steps 条消息（user/assistant 交替，以 assistant 结尾）"""
    need = 2 * prefix_steps
    if len(conv) < need:
        return None
    msgs = []
    for i, s in enumerate(conv[:need]):
        try:
            role = ROLES[s['from']]
        except KeyError:
            return None
        if role != ('user' if i % 2 == 0 else 'assistant'):
            return None
        msgs.append({'role': role, 'content': s['value']})
    if msgs[-1]['role'] != 'assistant':
        return None
    return msgs


def stage_cont(model, tok, trajs, args, out_path):
    done = {(r['qid'], r['traj_idx'], r['prefix_steps'], r['cont_idx']) for r in read_jsonl(out_path)}
    log('stage cont: 已完成 %d 次，开始补充' % len(done))

    # 先 enumerate 所有 (traj, prefix) 组合
    jobs = []
    for t in trajs:
        for ps in args.prefixes:
            msgs = prefix_messages(t['conversations'], ps)
            if msgs is None:
                continue
            jobs.append((t, ps, msgs))

    log('stage cont: 共 %d 个 (traj,prefix) 组合，每组合 %d 次 continuation' % (len(jobs), args.n_cont))

    for t, ps, base_msgs in jobs:
        # 记录前缀的 token 长度（用于控制 token position）
        from eval_agent.qa_rollout_chunked_context import _render_messages
        try:
            prompt = _render_messages(tok, base_msgs)
            ntok = len(tok(prompt, add_special_tokens=False).input_ids)
        except Exception:
            ntok = -1

        for ci in range(args.n_cont):
            if (t['qid'], t['traj_idx'], ps, ci) in done:
                continue
            # 深拷贝：generate_step 会就地 del messages[1:3]
            messages = [dict(m) for m in base_msgs]
            chunks = []
            t0 = time.time()
            for _ in range(args.cont_budget):
                chunk = generate_step(model, tok, messages, args.step_tokens,
                                      args.temperature, args.context_max_tokens)
                if not chunk:
                    break
                chunks.append(chunk)
                messages.append({'role': 'assistant', 'content': chunk})
                if has_explicit_final_marker(chunk):
                    break
                messages.append({'role': 'user', 'content': CONT_PROMPT})

            text = '\n'.join(chunks)
            normalized, fallback_pred, fin = finalize_rollout(text, 'strict')
            steps2 = split_steps(normalized)
            if fin and not has_explicit_final_marker('\n'.join(steps2)):
                fin = False
            pred2 = steps2[-1] if steps2 else text
            ok = bool(fin and (answers_equal(pred2, t['gold_answer']) or answers_equal(text, t['gold_answer'])))

            rec = {
                'qid': t['qid'], 'traj_idx': t['traj_idx'],
                'prefix_steps': ps, 'prefix_ntok': ntok,
                'cont_idx': ci,
                'n_generated': len(chunks),
                'finished': bool(fin), 'success': ok,
                'cont_text': text[:2000],
                'secs': round(time.time() - t0, 1),
            }
            append_jsonl(out_path, rec)
            log('  cont %s#%d ps=%d ntok=%d c=%d gen=%d ok=%s (%.1fs)' % (
                t['qid'], t['traj_idx'], ps, ntok, ci, len(chunks), ok, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['traj', 'cont', 'all'], default='all')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--input', default=INPUT)
    ap.add_argument('--outdir', default=str(ROOT / 'drift_exp/branch_pilot'))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--n_questions', type=int, default=40)
    ap.add_argument('--n_traj', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=128)
    ap.add_argument('--step_tokens', type=int, default=1024)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--context_max_tokens', type=int, default=32768)
    ap.add_argument('--wall_clock_timeout', type=float, default=1200.0)
    ap.add_argument('--prefixes', type=int, nargs='+', default=[8, 32, 64])
    ap.add_argument('--cont_budget', type=int, default=8,
                    help='统一续跑预算（所有前缀相同，杜绝位置-预算混淆）')
    ap.add_argument('--n_cont', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed + args.shard)

    rows = read_jsonl(args.input)
    if not rows:
        # 若 jsonl 不存在，退回 parquet
        raise SystemExit('input not found: %s' % args.input)
    rows = rows[:args.n_questions]
    # 按题目分片，保证同题的所有轨迹在同一 shard
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    log('shard %d/%d: %d 道题, n_traj=%d, prefixes=%s, cont_budget=%d, n_cont=%d' % (
        args.shard, args.nshards, len(rows), args.n_traj, args.prefixes, args.cont_budget, args.n_cont))

    traj_path = os.path.join(args.outdir, 'traj_shard%d.jsonl' % args.shard)
    cont_path = os.path.join(args.outdir, 'cont_shard%d.jsonl' % args.shard)

    model, tok = load_model(args.model)
    log('model loaded')

    if args.mode in ('traj', 'all'):
        stage_traj(model, tok, rows, args, traj_path)

    if args.mode in ('cont', 'all'):
        trajs = read_jsonl(traj_path)
        stage_cont(model, tok, trajs, args, cont_path)

    log('done. traj=%s cont=%s' % (traj_path, cont_path))


if __name__ == '__main__':
    main()
