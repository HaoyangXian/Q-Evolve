#!/usr/bin/env python3
"""
补充 critic 打分：对 branch_pilot 生成的轨迹前缀做 V/Qmin 打分。

branch_pilot 只做了「轨迹生成 + continuation + success 记录」，缺 critic 打分。
本脚本补齐：从 traj 的 conversations 截取前缀，用冻结 critic 打分，
输出 (qid, traj_idx, prefix_steps, prefix_ntok, V, Qmin)。

用法（用空闲卡，如卡4）：
  CUDA_VISIBLE_DEVICES=4 python score_prefixes.py \
    --traj drift_exp/branch_pilot/traj_shard0.jsonl \
    --prefixes 16 32 --out drift_exp/branch_pilot/prefix_scores_shard0.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QEvolve"))

CRITIC_BASE = str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success")
CRITIC_HEAD = str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B")


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def prefix_messages(conv, prefix_steps):
    """取前 2*prefix_steps 条消息（与 branch_pilot.prefix_messages 一致）"""
    need = 2 * prefix_steps
    if len(conv) < need:
        return None
    return conv[:need]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--prefixes", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_length", type=int, default=32768)
    args = ap.parse_args()

    trajs = read_jsonl(args.traj)

    # 1) 截取所有前缀 -> critic 输入 jsonl
    prefix_rows = []
    meta = []  # 与 prefix_rows 一一对应
    for t in trajs:
        conv = t.get("conversations") or []
        for ps in args.prefixes:
            msgs = prefix_messages(conv, ps)
            if msgs is None:
                continue
            prefix_rows.append({
                "id": f"{t['qid']}#{t['traj_idx']}::ps{ps}",
                "conversations": msgs,
                "question": t.get("question", ""),
                "gold_answer": t.get("gold_answer", ""),
                "assigned_reward": [0.0] * ps,
                "meta": {"success": False},
            })
            meta.append((t["qid"], t["traj_idx"], ps, len(msgs)))

    if not prefix_rows:
        print("[score] 无前缀可打分")
        return

    tmp = Path(args.out).with_suffix(".tmp.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in prefix_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2) 调用现有 critic 打分管线
    py = "/data3/xhy/miniconda3/envs/pettingllms/bin/python"
    exp_name = "branch_prefix_" + Path(args.traj).stem
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/QEvolve", TOKENIZERS_PARALLELISM="false")
    import subprocess
    cmd = [
        py, "-u", str(ROOT / "QEvolve/inf_critic_chunked_fast.py"),
        "--data_path", str(tmp),
        "--task", "qa", "--no_aux_reward",
        "--advantage_base_model", CRITIC_BASE,
        "--advantage_model_path", CRITIC_HEAD,
        "--max_length", str(args.max_length),
        "--exp_name", exp_name, "--strict",
    ]
    print("[score] 调用:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)

    # 3) 读取 critic 输出，映射回前缀
    adv_path = ROOT / "advantage_inference/qa_traj_tokens/wo_assigned_rewards" / exp_name / \
        f"adv_token_pred-gamma0.95-len32768-{exp_name}.jsonl"
    adv = read_jsonl(adv_path)

    results = []
    for i, a in enumerate(adv):
        if i >= len(meta):
            break
        qid, traj_idx, ps, nmsg = meta[i]
        V = a.get("V") or []
        Q = a.get("Qmin") or []
        if not V:
            continue
        results.append({
            "qid": qid, "traj_idx": traj_idx, "prefix_steps": ps,
            "V": float(V[-1]),       # 前缀最后一步 V
            "Qmin": float(Q[-1]),    # 前缀最后一步 Qmin
            "ep_len": a.get("ep_len"),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[score] 打分 {len(results)} 个前缀 -> {args.out}")


if __name__ == "__main__":
    main()
