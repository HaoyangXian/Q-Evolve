#!/usr/bin/env python3
"""
能力矩阵聚合 + 维度可辨识性分析。

1. 聚合 3 档 checkpoint 的 per-subfield 成功率 -> capability_matrix.json
2. 维度可辨识性：3 档之间各子领域的 acc 变化是否"分化"（不同维度不同变化）
   还是"同构"（所有维度一起涨）

若同构 -> 现有 checkpoint 不足以支撑多维诊断，需补充不同能力配置的 checkpoint。

用法：python analyze_matrix.py
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")
OUT = ROOT / "drift_exp/capability_probe/results"
PROBE = ROOT / "drift_exp/capability_probe/probe_bank/probe_pool.jsonl"

CHECKPOINTS = ["base", "sft", "policy"]


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def main():
    # probe 元信息
    subfield_of = {}
    for r in read_jsonl(PROBE):
        subfield_of[str(r["id"])] = r.get("subfield", "Other")

    # 各 checkpoint per-subfield acc
    matrix = {}
    for ck in CHECKPOINTS:
        rows = read_jsonl(OUT / f"rollout_{ck}.jsonl")
        succ = {}
        for r in rows:
            rid = str(r.get("id", "")).split("::")[0]
            succ[rid] = bool(r.get("meta", {}).get("success", False))
        by_sub = defaultdict(lambda: [0, 0])
        for rid, sub in subfield_of.items():
            by_sub[sub][1] += 1
            by_sub[sub][0] += 1 if succ.get(rid, False) else 0
        matrix[ck] = {
            "n": len(rows),
            "acc": sum(succ.values()) / max(len(succ), 1),
            "by_subfield": {k: {"correct": v[0], "total": v[1], "acc": v[0] / max(v[1], 1)}
                            for k, v in by_sub.items()},
        }

    (OUT / "capability_matrix.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2))

    subfields = sorted(set(subfield_of.values()))
    print("=== 能力响应矩阵（per-subfield acc）===")
    header = "checkpoint".ljust(11) + "".join(s.ljust(13) for s in subfields) + "overall"
    print(header)
    for ck in CHECKPOINTS:
        m = matrix[ck]
        row = ck.ljust(11)
        for s in subfields:
            b = m["by_subfield"].get(s)
            row += (f"{b['acc']:.3f}" if b else "-").ljust(13)
        row += f"{m['acc']:.3f}"
        print(row)

    # 维度分化分析：每档相对 base 的 Δ
    print("\n=== 维度分化分析（相对 base 的 acc 变化 Δ）===")
    base_sub = matrix["base"]["by_subfield"]
    for ck in ["sft", "policy"]:
        ck_sub = matrix[ck]["by_subfield"]
        deltas = {}
        for s in subfields:
            if s in base_sub and s in ck_sub:
                deltas[s] = ck_sub[s]["acc"] - base_sub[s]["acc"]
        spread = max(deltas.values()) - min(deltas.values()) if deltas else 0
        print(f"  {ck}: Δ = " + "  ".join(f"{s}={deltas[s]:+.3f}" for s in subfields if s in deltas))
        print(f"     维度分化度(spread) = {spread:.3f}  "
              f"{'→ 存在维度分化' if spread > 0.08 else '→ 各维度几乎同构（checkpoint 不足以支撑多维诊断）'}")

    print(f"\n[analyze_matrix] 能力矩阵 -> {OUT/'capability_matrix.json'}")


if __name__ == "__main__":
    main()
