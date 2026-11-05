#!/usr/bin/env python3
"""
Experiment 3: Termination Intervention (P2)
保持推理文本内容不变，只改变输入 Critic 时的「结束形式」，测量 V 的变化。

条件：
  E1_normal     : 标准渲染（最后 assistant 带 <|im_end|>\n）—— 基线
  E2_eos        : E1 + <|endoftext|>            （state_pos 固定=E1）
  E3_trunc      : 最后 assistant 的 suffix 去掉  （硬截断，state_pos 随之前移）
  E4_continue   : E1 + user "Continue..."       （state_pos 固定=E1）
  E5_ast_open   : E1 去 suffix + 重开 assistant  （state_pos 固定=E1）

理论预期（causal attention）：
  V(E2) == V(E4) == V(E5) == V(E1)   ← 后缀在 state_pos 之后，不应影响
  V(E3) 可能明显不同              ← 因为缺失 <|im_end|>，且 state_pos 前移
"""
import json, sys, os, pathlib
import numpy as np
import torch

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'QEvolve'))

from transformers import AutoTokenizer
from eval_agent.data_utils import get_chat_template
from inf_critic_chunked_fast import load_probe_full_state_dict

MODEL_BASE = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/sft_success')
MODEL_CKPT = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B')
MAX_LENGTH = 32768
N_QUESTIONS = 50
POSITIONS = [8, 32]

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---- 确定性映射 ----
det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

# ---- 加载模型 ----
print('[Load] 正在加载 critic...', flush=True)
model, tokenizer, resolved = load_probe_full_state_dict(
    base_model_name_or_path=MODEL_BASE,
    ckpt_dir=MODEL_CKPT,
    device='cuda:0',
    max_length=MAX_LENGTH,
    strict=True,
    use_done_head=False,
)
model.eval()
inner = model
tmpl = get_chat_template(MODEL_BASE)
print('[Load] done:', resolved, flush=True)

P_SUF = tmpl.role_prefix_and_suffix
EOS_TOKEN = tokenizer.eos_token or '<|endoftext|>'
CONT_MSG = "Continue with exactly one concise new reasoning step."

def build_ids(messages, drop_last_suffix=False, append_raw='', append_assistant_open=False):
    """增量 tokenize，返回 (ids, last_end_pos)
    last_end_pos = 最后一个 assistant 消息的末尾 token 索引（语义上的 state 位置）"""
    ids, offsets, acc = [], [], 0
    last_asst_end = None
    for i, m in enumerate(messages):
        pre, suf = P_SUF.get(m['role'], ('', ''))
        if drop_last_suffix and i == len(messages) - 1:
            suf = ''
        piece = pre + m['content'] + suf
        t = tokenizer(piece, add_special_tokens=False).input_ids
        ids.extend(t); acc += len(t)
        offsets.append(acc)
        if m['role'] == 'assistant':
            last_asst_end = acc - 1
    if append_raw:
        t = tokenizer(append_raw, add_special_tokens=False).input_ids
        ids.extend(t); acc += len(t)
    if append_assistant_open:
        t = tokenizer(P_SUF['assistant'][0], add_special_tokens=False).input_ids
        ids.extend(t); acc += len(t)
    return ids, last_asst_end

# ---- 构造样本 ----
samples = []
for d in det[:N_QUESTIONS]:
    item = solA[d['solver_row']]
    conv = item.get('conversations') or []
    for pos in POSITIONS:
        n_msgs_needed = 1 + 2 * pos  # user + (asst,user)*(pos-1)? 简化：取前 2*pos 条
        msgs = []
        ok = True
        for i, sent in enumerate(conv[:2 * pos]):
            try:
                r = roles[sent['from']]
            except KeyError:
                ok = False; break
            exp = 'user' if i % 2 == 0 else 'assistant'
            if r != exp:
                ok = False; break
            msgs.append({'role': r, 'content': sent['value']})
        if not ok or not msgs or msgs[-1]['role'] != 'assistant':
            continue
        samples.append({'qid': d['qid'], 'pos': pos, 'msgs': msgs})

print(f'[Data] 构造 {len(samples)} 条前缀 (题目×位置)', flush=True)

# ---- 构造 5 种条件 ----
CONDS = ['E1_normal', 'E2_eos', 'E3_trunc', 'E4_continue', 'E5_ast_open']

def make_cond(s, cond):
    m = s['msgs']
    if cond == 'E1_normal':
        return build_ids(m)
    if cond == 'E2_eos':
        return build_ids(m, append_raw=EOS_TOKEN)
    if cond == 'E3_trunc':
        return build_ids(m, drop_last_suffix=True)
    if cond == 'E4_continue':
        m2 = m + [{'role': 'user', 'content': CONT_MSG}]
        # 需要以 user 结尾，但 state_pos 仍是最后 assistant 的位置
        ids, last_end = build_ids(m2)
        # 重新计算最后 assistant 的 end
        _, last_end = build_ids(m)
        return ids, last_end
    if cond == 'E5_ast_open':
        return build_ids(m, drop_last_suffix=True, append_assistant_open=True)
    raise ValueError(cond)

# ---- 推理 ----
records = []
for cond in CONDS:
    for s in samples:
        ids, spos = make_cond(s, cond)
        if spos is None or spos >= MAX_LENGTH or len(ids) > MAX_LENGTH:
            continue
        records.append({'qid': s['qid'], 'pos': s['pos'], 'cond': cond,
                        'ids': ids, 'spos': spos})

print(f'[Data] 共 {len(records)} 条待推理', flush=True)

def run_batch(recs, device='cuda:0'):
    B = len(recs)
    maxL = max(len(r['ids']) for r in recs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.full((B, maxL), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, maxL), dtype=torch.long, device=device)
    state_pos = torch.zeros((B, 1), dtype=torch.long, device=device)
    for i, r in enumerate(recs):
        L = len(r['ids'])
        input_ids[i, :L] = torch.tensor(r['ids'], dtype=torch.long, device=device)
        attn[i, :L] = 1
        state_pos[i, 0] = r['spos']
    with torch.no_grad():
        out = inner.forward_v_multi(
            input_ids=input_ids, attention_mask=attn,
            state_pos=state_pos, next_pos=state_pos,
        )
        V = out['V'].float().squeeze(-1)[:, 0]
    return V.detach().cpu().numpy()

# 按长度分批
records.sort(key=lambda r: len(r['ids']))
BATCH = 8
vals = {}
for i in range(0, len(records), BATCH):
    batch = records[i:i+BATCH]
    v = run_batch(batch)
    for r, vv in zip(batch, v):
        vals[(r['qid'], r['pos'], r['cond'])] = float(vv)
    if (i // BATCH) % 5 == 0:
        print(f'  {i+len(batch)}/{len(records)}', flush=True)

print(f'[Done] 得到 {len(vals)} 个 V 值', flush=True)

# ---- 分析 ----
print()
print('=' * 78)
print('Experiment 3: Termination Intervention 结果')
print('=' * 78)

keys = [(q, p) for (q, p, c) in vals if c == 'E1_normal']
print(f'有效 (题目,位置) 组合: {len(keys)}')
print()
print('各条件 V 的均值（全部组合）:')
for cond in CONDS:
    arr = [vals[(q, p, cond)] for (q, p) in keys if (q, p, cond) in vals]
    print('  %-14s mean=%+.5f  median=%+.5f  n=%d' % (cond, np.mean(arr), np.median(arr), len(arr)))

print()
print('相对 E1 的差异（同题目同位置配对）:')
print('%-14s %12s %12s %12s %10s' % ('条件', 'mean ΔV', 'median ΔV', 'max|ΔV|', '|ΔV|<1e-6'))
for cond in CONDS[1:]:
    dv = []
    for (q, p) in keys:
        k1 = (q, p, 'E1_normal'); k2 = (q, p, cond)
        if k1 in vals and k2 in vals:
            dv.append(vals[k2] - vals[k1])
    dv = np.array(dv)
    if len(dv) == 0: continue
    print('%-14s %+12.5f %+12.5f %12.5f %9.1f%%' % (
        cond, dv.mean(), np.median(dv), np.abs(dv).max(), 100*np.mean(np.abs(dv) < 1e-6)))

print()
print('按位置分层（mean ΔV vs E1）:')
print('%-14s %12s %12s' % ('条件', 'pos=8', 'pos=32'))
for cond in CONDS[1:]:
    row = []
    for p in POSITIONS:
        dv = [vals[(q, pp, cond)] - vals[(q, pp, 'E1_normal')]
              for (q, pp) in keys if pp == p and (q, pp, cond) in vals]
        row.append(np.mean(dv) if dv else float('nan'))
    print('%-14s %+12.5f %+12.5f' % (cond, row[0], row[1]))

# ---- 判定 ----
print()
print('=' * 78)
print('判定')
print('=' * 78)
for cond in ['E2_eos', 'E4_continue', 'E5_ast_open']:
    dv = np.array([vals[(q, p, cond)] - vals[(q, p, 'E1_normal')]
                   for (q, p) in keys if (q, p, cond) in vals])
    if len(dv):
        flag = '✓ 一致（causal 自洽）' if np.abs(dv).max() < 1e-4 else '✗ 不一致（异常！）'
        print('  %-14s max|ΔV|=%.2e  %s' % (cond, np.abs(dv).max(), flag))

dv_trunc = np.array([vals[(q, p, 'E3_trunc')] - vals[(q, p, 'E1_normal')]
                     for (q, p) in keys if (q, p, 'E3_trunc') in vals])
if len(dv_trunc):
    print()
    print('  E3_trunc（硬截断 vs 正常结束）:')
    print('    mean ΔV = %+.5f   median = %+.5f   max|ΔV| = %.5f' % (
        dv_trunc.mean(), np.median(dv_trunc), np.abs(dv_trunc).max()))
    # 与 A 档内部 V 的变异尺度对比
    base_v = np.array([vals[(q, p, 'E1_normal')] for (q, p) in keys])
    print('    基线 V 的 std = %.5f  →  ΔV / std = %.2f' % (
        base_v.std(), abs(dv_trunc.mean()) / max(base_v.std(), 1e-9)))
    print('    ΔV 显著 (>10%%V的std) 的题目比例: %.1f%%' % (
        100*np.mean(np.abs(dv_trunc) > 0.1*base_v.std())))

# 保存
outp = ROOT / 'drift_exp' / 'exp3_termination_results.jsonl'
with open(outp, 'w', encoding='utf-8') as f:
    for (q, p, c), v in vals.items():
        f.write(json.dumps({'qid': q, 'pos': p, 'cond': c, 'V': v}) + '\n')
print()
print('结果已保存:', outp)
