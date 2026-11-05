#!/usr/bin/env python3
"""
能力矩阵评测：多 checkpoint × 多 Probe 跑成功率，输出能力响应矩阵。

复用 eval_agent/qa_rollout.py（subprocess 调用），分三阶段：
  --prepare   生成每个 checkpoint 的评测 input（probe_pool.jsonl 已可直接用）
  --run       subprocess 调 qa_rollout.py（指定 GPU 并行）
  --aggregate 聚合成功率 -> capability_matrix.json

用法：
  python eval_capability.py --prepare --probe drift_exp/capability_probe/probe_bank/probe_pool.jsonl
  python eval_capability.py --run --checkpoint sft --gpu 0
  python eval_capability.py --aggregate
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
PY = "/data3/xhy/miniconda3/envs/pettingllms/bin/python"
ROLLOUT = ROOT / "eval_agent/qa_rollout.py"
OUTDIR = ROOT / "drift_exp/capability_probe/results"

# checkpoint 配置：model = 完整模型路径，adapter = LoRA（可选）
# 注意：policy 的 base 是 sft_success（RL 在 SFT 基础上训练），adapter 是 RL LoRA
CHECKPOINTS = {
    "base": {
        "model": "/data3/xhy/codes/models/Qwen3-8B",
        "adapter": None,
    },
    "sft": {
        "model": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success"),
        "adapter": None,
    },
    "policy": {
        "model": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/sft_success"),
        "adapter": str(ROOT / "runs/olympiad_qwen3_8b_iterative1024_small/policy_chunked8192"),
    },
}

# 评测参数（temperature=0 保证确定性，与 D1 一致）
EVAL_KWARGS = {
    "samples": 1,
    "max_new_tokens": 8192,
    "step_max_new_tokens": 1024,
    "max_steps": 96,
    "temperature": 0,
    "termination_mode": "strict",
}


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def prepare(args):
    probes = read_jsonl(args.probe)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # 生成统一 input（qa_rollout 用 id/question/final_answer）
    inp = OUTDIR / "probe_input.jsonl"
    with open(inp, "w", encoding="utf-8") as f:
        for r in probes:
            f.write(json.dumps({
                "id": r["id"], "question": r["question"],
                "final_answer": r["final_answer"],
            }, ensure_ascii=False) + "\n")
    print(f"[prepare] 评测 input: {len(probes)} 题 -> {inp}")


def run(args):
    ckpt = CHECKPOINTS[args.checkpoint]
    inp = OUTDIR / "probe_input.jsonl"
    out = OUTDIR / f"rollout_{args.checkpoint}.jsonl"
    log = OUTDIR / f"rollout_{args.checkpoint}.log"

    cmd = [PY, "-u", str(ROLLOUT),
           "--input", str(inp), "--output", str(out),
           "--model", ckpt["model"]]
    if ckpt["adapter"]:
        cmd += ["--adapter", ckpt["adapter"]]
    for k, v in EVAL_KWARGS.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]

    env = dict(os.environ,
               PYTHONPATH=f"{ROOT}:{ROOT}/QEvolve",
               CUDA_VISIBLE_DEVICES=str(args.gpu),
               TOKENIZERS_PARALLELISM="false")
    print(f"[run] {args.checkpoint} 启动 (GPU {args.gpu})")
    with open(log, "w") as lf:
        subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[run] {args.checkpoint} 完成 -> {out}")


def aggregate(args):
    probes = read_jsonl(OUTDIR / "probe_input.jsonl")
    # probe id -> (subfield, probe_type)
    pool_info = {}
    pool_path = ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"
    for r in read_jsonl(pool_path):
        pool_info[str(r["id"])] = (r.get("subfield"), r.get("probe_type"))

    matrix = {}
    for name in CHECKPOINTS:
        out = OUTDIR / f"rollout_{name}.jsonl"
        rows = read_jsonl(out)
        # id -> success
        succ = {}
        for r in rows:
            rid = str(r.get("id", "")).split("::")[0]
            succ[rid] = bool(r.get("meta", {}).get("success", False))
        # 按 subfield 聚合
        by_sub = defaultdict(lambda: [0, 0])
        for r in probes:
            rid = str(r["id"])
            sub, _ = pool_info.get(rid, ("Other", None))
            by_sub[sub][1] += 1
            by_sub[sub][0] += 1 if succ.get(rid, False) else 0
        matrix[name] = {
            "n_total": len(rows),
            "acc": sum(succ.values()) / max(len(succ), 1),
            "by_subfield": {k: {"correct": v[0], "total": v[1],
                                "acc": v[0] / max(v[1], 1)} for k, v in by_sub.items()},
        }

    out_path = OUTDIR / "capability_matrix.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)

    # 打印能力矩阵
    print("\n=== 能力响应矩阵（by subfield accuracy）===")
    subfields = sorted(set(s for m in matrix.values() for s in m["by_subfield"]))
    header = "checkpoint".ljust(10) + "".join(s.ljust(14) for s in subfields) + "overall"
    print(header)
    for name, m in matrix.items():
        row = name.ljust(10)
        for s in subfields:
            b = m["by_subfield"].get(s)
            row += (f"{b['acc']:.3f}" if b else "-").ljust(14)
        row += f"{m['acc']:.3f}"
        print(row)
    print(f"\n[aggregate] 能力矩阵 -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--probe", default=str(ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"))
    ap.add_argument("--checkpoint", choices=list(CHECKPOINTS.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.prepare:
        prepare(args)
    if args.run:
        run(args)
    if args.aggregate:
        aggregate(args)
    if not (args.prepare or args.run or args.aggregate):
        ap.print_help()


if __name__ == "__main__":
    main()
