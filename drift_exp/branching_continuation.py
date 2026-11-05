#!/usr/bin/env python3
"""
Branching Continuation Experiment (D1.1-B / P1)
================================================
核心科学问题：同一题、同一推理位置、相近 token 长度下，
Critic 的 V / Qmin 能否预测「该状态未来的成功概率」？
（区分 H1 质量估计 vs H2 位置效应）

设计（采纳审稿建议）：
  - 每题采样 3 条独立轨迹（temperature 非零，不同 seed）
  - 在每条轨迹的 step 8 / 32 / 64 处截取前缀
  - 对每个前缀用【冻结 Critic】打分，取该位置最后一步的 V / Qmin
  - 对每个前缀用【同一 Solver】续采样 8 条 continuation（vLLM prefix caching 复用前缀 KV）
  - 记录每条 continuation 是否成功（meta.success）
  - 分析：同位置下 V/Qmin 与后续成功率的 Spearman/Pearson + 校准曲线

关键控制（审稿强调）：
  C1 固定剩余预算：所有 continuation 用统一 max_steps_remaining，不因前缀位置不同而不同
  C2 控制 token position：记录每个前缀的实际 token 长度，作为协变量

vLLM prefix caching：同一前缀的 8 条 continuation 共享前缀 KV（enable_prefix_caching=True）

用法（三阶段，可分开跑）：
  python branching_continuation.py --stage 1 --n_questions 40 --n_trajs 3 --positions 8,32,64 --temperature 0.8
  python branching_continuation.py --stage 2   # 对前缀打分
  python branching_continuation.py --stage 3   # 续采样 + 分析
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QEvolve"))

# 冻结资源（Branching 全程固定）
SOLVER_MODEL = "/data3/xhy/codes/models/Qwen3-8B"
CRITIC_BASE = str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success")
CRITIC_HEAD = str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B")
INPUT_JSONL = str(ROOT / "data/qa/olympiadbench_eval200_seed20261031.jsonl")
WORK_DIR = str(ROOT / "drift_exp" / "branching")


def load_rows(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


# ---------------------------------------------------------------------------
# Stage 1: 采样多条独立轨迹，提取中间前缀
# ---------------------------------------------------------------------------
def stage1_generate(args):
    """每题采样 n_trajs 条独立轨迹（temperature 非零），提取 step 前缀。"""
    from vllm import LLM, SamplingParams
    from eval_agent.qa_rollout_vllm_batched import (
        get_question, get_gold, initial_messages, build_prompt_tokens,
    )

    os.makedirs(WORK_DIR, exist_ok=True)
    rows = load_rows(INPUT_JSONL)[: args.n_questions]

    llm = LLM(
        model=SOLVER_MODEL,
        enable_prefix_caching=True,      # 关键：启用前缀 KV 复用
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        enforce_eager=False,
    )
    tok = llm.get_tokenizer()

    positions = [int(x) for x in args.positions.split(",")]

    all_prefixes = []
    for qi, row in enumerate(rows):
        question = get_question(row)
        gold = get_gold(row)
        for traj_i in range(args.n_trajs):
            # 用不同 seed 生成独立轨迹
            sp = SamplingParams(
                temperature=args.temperature,
                top_p=0.9,
                max_tokens=args.step_max_new_tokens,
                seed=args.base_seed + qi * 100 + traj_i,
            )
            msgs = initial_messages(question)
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            # 第一步生成
            out = llm.generate(prompt_token_ids=[prompt], sampling_params=sp)[0]
            first_step = out.outputs[0].text

            # 用迭代式逐步生成到最远位置（每步一次 generate，前缀自动复用）
            from eval_agent.qa_rollout_vllm_batched import has_explicit_final_marker
            steps = [first_step]
            conv = list(msgs) + [{"role": "assistant", "content": first_step}]
            max_pos = max(positions)
            while len(steps) < max_pos and not has_explicit_final_marker(steps[-1]):
                p = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
                o = llm.generate(prompt_token_ids=[p], sampling_params=sp)[0]
                chunk = o.outputs[0].text
                if not chunk:
                    break
                steps.append(chunk)
                conv.append({"role": "assistant", "content": chunk})

            # 提取每个位置的前缀（conversations 前缀 = 前 2*t 条消息 + 最后一条 assistant）
            for pos in positions:
                if pos > len(steps):
                    continue
                # step pos 的前缀：含前 pos 个 assistant 输出
                prefix_conv = list(msgs)
                for s in steps[:pos]:
                    prefix_conv.append({"role": "assistant", "content": s})
                # token 长度（供 C2 控制）
                prefix_ids = tok.apply_chat_template(prefix_conv, add_generation_prompt=False, tokenize=True)
                all_prefixes.append({
                    "qid": row.get("id", f"q{qi}"),
                    "question": question,
                    "gold": gold,
                    "traj_i": traj_i,
                    "pos": pos,
                    "prefix_conv": prefix_conv,
                    "prefix_tokens": len(prefix_ids),
                })

    out = Path(WORK_DIR) / "prefixes.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in all_prefixes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[stage1] 生成 {len(all_prefixes)} 个前缀 -> {out}")


# ---------------------------------------------------------------------------
# Stage 2: 对每个前缀用冻结 Critic 打分（V / Qmin）
# ---------------------------------------------------------------------------
def stage2_score(args):
    """对每个前缀打分。

    实现策略：把前缀写成完整 trajectory jsonl（含 conversations），
    复用现有的 critic 打分管线（preprocess_traj_dataset_qa_fast + 前向），
    这样 step 边界（state_pos/action_pos/next_pos）与训练时完全一致，
    避免手写前向引入的序列化差异。
    """
    import subprocess

    # 1) 前缀 -> trajectory jsonl（critic 输入格式）
    prefixes = [json.loads(x) for x in open(Path(WORK_DIR) / "prefixes.jsonl") if x.strip()]
    traj_path = Path(WORK_DIR) / "prefixes_traj.jsonl"
    with open(traj_path, "w", encoding="utf-8") as f:
        for p in prefixes:
            f.write(json.dumps({
                "id": f"{p['qid']}::branch{p['traj_i']}::pos{p['pos']}",
                "conversations": p["prefix_conv"],
                "question": p["question"],
                "gold_answer": p["gold"],
                "assigned_reward": [0.0] * (p["pos"]),
                "meta": {"success": False},
            }, ensure_ascii=False) + "\n")

    # 2) 调用现有 critic 打分脚本（冻结 critic，与 D1 完全一致）
    py = "/data3/xhy/miniconda3/envs/pettingllms/bin/python"
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/QEvolve", TOKENIZERS_PARALLELISM="false")
    cmd = [
        py, "-u", str(ROOT / "QEvolve/inf_critic_chunked_fast.py"),
        "--data_path", str(traj_path),
        "--task", "qa", "--no_aux_reward",
        "--advantage_base_model", CRITIC_BASE,
        "--advantage_model_path", CRITIC_HEAD,
        "--max_length", str(args.max_length),
        "--exp_name", "branching_prefix", "--strict",
    ]
    print("[stage2] 调用 critic 打分:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    # 3) 读取 critic 输出，映射回 (qid, traj_i, pos)
    # 前缀较短（<=64 步），一般不会超 32768 token，故 traj_id == 输入行序。
    # 若确有超长被过滤，需用 ep_len 序列对齐（参照 deterministic_verify 的做法）。
    adv_path = ROOT / "advantage_inference/qa_traj_tokens/wo_assigned_rewards/branching_prefix/adv_token_pred-gamma0.95-len32768-branching_prefix.jsonl"
    adv = [json.loads(x) for x in open(adv_path) if x.strip()]
    results = []
    for i, a in enumerate(adv):
        if i >= len(prefixes):
            break
        p = prefixes[i]
        V = a.get("V") or []; Q = a.get("Qmin") or []
        if not V:
            continue
        results.append({
            "qid": p["qid"], "traj_i": p["traj_i"], "pos": p["pos"],
            "prefix_tokens": p["prefix_tokens"],
            "V": float(V[-1]),          # 前缀最后一步的 V
            "Qmin": float(Q[-1]),       # 前缀最后一步的 Qmin
        })

    out = Path(WORK_DIR) / "prefix_scores.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage2] 打分 {len(results)} 个前缀 -> {out}")


# ---------------------------------------------------------------------------
# Stage 3: 从每个前缀续采样 8 条 continuation + 分析
# ---------------------------------------------------------------------------
def stage3_continue_and_analyze(args):
    from vllm import LLM, SamplingParams
    from eval_agent.qa_verifier import answers_equal, has_explicit_final_marker

    llm = LLM(
        model=SOLVER_MODEL,
        enable_prefix_caching=True,   # 同一前缀的 8 条 continuation 共享前缀 KV
        gpu_memory_utilization=0.85,
        max_model_len=32768,
    )
    tok = llm.get_tokenizer()

    prefixes = [json.loads(x) for x in open(Path(WORK_DIR) / "prefixes.jsonl") if x.strip()]
    scores = { (r["qid"], r["traj_i"], r["pos"]): r for r in
               [json.loads(x) for x in open(Path(WORK_DIR) / "prefix_scores.jsonl") if x.strip()] }

    # C1: 统一剩余预算（不因前缀位置不同而不同）
    remain_steps = args.max_steps_remaining

    records = []
    for p in prefixes:
        qid, traj_i, pos = p["qid"], p["traj_i"], p["pos"]
        sc = scores.get((qid, traj_i, pos))
        # 同一前缀 8 条 continuation：相同 prompt，不同 seed
        succ = []
        for c in range(args.n_continuations):
            sp = SamplingParams(
                temperature=args.temperature,
                top_p=0.9,
                max_tokens=args.step_max_new_tokens,
                seed=args.base_seed + 1000 * pos + c,
            )
            conv = list(p["prefix_conv"])
            done = False
            for _ in range(remain_steps):
                prompt = tok.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
                o = llm.generate(prompt_token_ids=[prompt], sampling_params=sp)[0]
                chunk = o.outputs[0].text
                if not chunk:
                    break
                conv.append({"role": "assistant", "content": chunk})
                if has_explicit_final_marker(chunk):
                    done = True
                    break
            # 判定成功：最终答案 == gold
            pred = conv[-1]["content"] if conv else ""
            succ.append(1.0 if answers_equal(pred, p["gold"]) else 0.0)
        success_rate = np.mean(succ)
        records.append({
            "qid": qid, "traj_i": traj_i, "pos": pos,
            "V": sc["V"] if sc else None,
            "Qmin": sc["Qmin"] if sc else None,
            "prefix_tokens": p["prefix_tokens"],
            "success_rate": float(success_rate),
            "n_succ": int(sum(succ)), "n_cont": len(succ),
        })

    out = Path(WORK_DIR) / "branching_results.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 分析：同位置下 V/Qmin 与成功率的关联 ----------
    print("\n" + "=" * 72)
    print("分析：同位置下 Critic 分数 vs 后续成功率")
    print("=" * 72)
    from scipy.stats import spearmanr, pearsonr

    for pos in sorted(set(r["pos"] for r in records)):
        sub = [r for r in records if r["pos"] == pos and r["V"] is not None]
        if len(sub) < 5:
            continue
        V = [r["V"] for r in sub]; Q = [r["Qmin"] for r in sub]
        rate = [r["success_rate"] for r in sub]
        toklen = [r["prefix_tokens"] for r in sub]
        rho_v, p_v = spearmanr(V, rate)
        rho_q, p_q = spearmanr(Q, rate)
        # C2: 控制 token 长度的偏相关（残差化后相关）
        def partial_corr(x, y, z):
            for v in (x, y, z):
                pass
            # 简单做法：线性去 z 后相关
            def resid(a, c):
                c = np.asarray(c, float)
                coef = np.polyfit(c, a, 1)
                return np.asarray(a, float) - (coef[0] * c + coef[1])
            rx, ry = resid(x, z), resid(y, z)
            return pearsonr(rx, ry)[0]
        pc_v = partial_corr(V, rate, toklen)
        pc_q = partial_corr(Q, rate, toklen)
        print(f"  pos={pos:3d} (n={len(sub)}):")
        print(f"    V   Spearman={rho_v:+.3f} (p={p_v:.3f})  控制token长度偏相关={pc_v:+.3f}")
        print(f"    Qmin Spearman={rho_q:+.3f} (p={p_q:.3f})  控制token长度偏相关={pc_q:+.3f}")

    print(f"\n[stage3] 结果 -> {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--n_questions", type=int, default=40)
    ap.add_argument("--n_trajs", type=int, default=3)
    ap.add_argument("--positions", type=str, default="8,32,64")
    ap.add_argument("--n_continuations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--step_max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_steps_remaining", type=int, default=96, help="C1: 统一剩余预算")
    ap.add_argument("--max_length", type=int, default=32768)
    ap.add_argument("--base_seed", type=int, default=20261104)
    args = ap.parse_args()

    if args.stage == 1:
        stage1_generate(args)
    elif args.stage == 2:
        stage2_score(args)
    else:
        stage3_continue_and_analyze(args)


if __name__ == "__main__":
    main()
