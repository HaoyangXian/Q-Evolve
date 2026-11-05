#!/usr/bin/env python3
"""
Budget 敏感性维度分解：不同推理预算（max_steps）下，各子领域能力如何退化。

核心问题：budget 减少时，是「所有维度一起退化」，还是「某些维度先退化」？
若不同维度有不同 budget 敏感性（如 planning 先退化、concept 后退化），
说明「推理预算」本身对不同能力维度有差异化影响——这是多维诊断的重要混杂/信号。

用法：python budget_analysis.py
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
OUT = ROOT / "drift_exp/capability_probe/results"
PROBE = ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"

# budget 档位（8/16/32/64/96）
BUDGETS = [8, 16, 32, 64, 96]


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def acc_by_sub(rollout_path, subfield_of):
    succ = {}
    for r in read_jsonl(rollout_path):
        rid = str(r.get("id", "")).split("::")[0]
        succ[rid] = bool(r.get("meta", {}).get("success", False))
    by_sub = defaultdict(lambda: [0, 0])
    for rid, sub in subfield_of.items():
        by_sub[sub][1] += 1
        by_sub[sub][0] += 1 if succ.get(rid, False) else 0
    return {k: v[0] / max(v[1], 1) for k, v in by_sub.items()}


def main():
    subfield_of = {str(r["id"]): r.get("subfield", "Other") for r in read_jsonl(PROBE)}
    subfields = sorted(set(subfield_of.values()))

    # 每档 budget 的 per-subfield acc
    budget_acc = {}
    for b in BUDGETS:
        if b == 96:
            path = OUT / "rollout_sft.jsonl"  # cap_sft 即 budget=96
        else:
            path = OUT / f"rollout_sft_b{b}.jsonl"
        budget_acc[b] = acc_by_sub(path, subfield_of)

    print("=== 各 subfield 成功率随 budget 变化（sft 模型）===")
    header = "budget".ljust(8) + "".join(s.ljust(13) for s in subfields)
    print(header)
    for b in BUDGETS:
        row = str(b).ljust(8)
        for s in subfields:
            a = budget_acc[b].get(s, 0.0)
            row += f"{a:.3f}".ljust(13)
        print(row)

    # budget 敏感性：从 8 到 96 的恢复幅度
    print("\n=== 各维度 budget 敏感性（b8 -> b96 的恢复幅度）===")
    for s in subfields:
        lo = budget_acc[8].get(s, 0.0)
        hi = budget_acc[96].get(s, 0.0)
        print(f"  {s}: {lo:.3f} -> {hi:.3f}  (Δ={hi-lo:+.3f})")

    # 判断：各维度恢复幅度是否差异大
    deltas = {s: budget_acc[96].get(s, 0) - budget_acc[8].get(s, 0) for s in subfields}
    spread = max(deltas.values()) - min(deltas.values())
    print(f"\n  维度间 Δ 的 spread = {spread:.3f}  "
          f"{'→ 不同维度对预算敏感性不同（budget 有维度化影响）' if spread > 0.08 else '→ 各维度对预算同等敏感（budget 是整体截断）'}")


if __name__ == "__main__":
    main()
