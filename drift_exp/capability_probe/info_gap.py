#!/usr/bin/env python3
"""
信息缺口验证（Identifiability）：被动观测是否信息不足？

输入 capability_matrix.json（各 checkpoint 在 per-subfield / per-probe 的成功率），
构造被动分布 q0（Balanced / Skewed），计算 checkpoint 两两之间的区分信息量（KL），
输出最坏情况信息量 I_min(q0)，判断是否存在「被动观测难区分」的状态对。

用法：
  python info_gap.py --matrix drift_exp/capability_probe/results/capability_matrix.json
"""
import argparse
import json
import math
from pathlib import Path
from itertools import combinations

ROOT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve")

# 被动分布（按科目采样比例）
PASSIVE_DIST = {
    "balanced": {"Algebra": 0.25, "Combinatorics": 0.25, "Geometry": 0.25, "Number Theory": 0.25},
    "skewed": {"Algebra": 0.55, "Combinatorics": 0.15, "Geometry": 0.15, "Number Theory": 0.15},
}


def bern_kl(p, q):
    """KL(Bernoulli(p) || Bernoulli(q))。"""
    p = min(max(p, 1e-6), 1 - 1e-6)
    q = min(max(q, 1e-6), 1 - 1e-6)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(ROOT / "drift_exp/capability_probe/results/capability_matrix.json"))
    args = ap.parse_args()

    matrix = json.load(open(args.matrix, encoding="utf-8"))
    names = list(matrix.keys())

    # 提取 per-subfield 成功率向量
    subfields = sorted(set(s for m in matrix.values() for s in m["by_subfield"]))
    theta = {n: {s: matrix[n]["by_subfield"].get(s, {}).get("acc", 0.0) for s in subfields}
             for n in names}

    print("=== 各 checkpoint 的能力签名（per-subfield acc）===")
    for n in names:
        sig = "  ".join(f"{s}={theta[n][s]:.3f}" for s in subfields)
        print(f"  {n}: {sig}")

    for dist_name, q0 in PASSIVE_DIST.items():
        print(f"\n=== 被动分布 {dist_name} ===")
        # 归一化
        tot = sum(q0.get(s, 0.0) for s in subfields)
        q0 = {s: q0.get(s, 0.0) / max(tot, 1e-6) for s in subfields}

        # 两两 KL 信息量
        print(f"  {'对':<18}{'KL 总信息量':>12}{'最弱维度 KL':>14}")
        pair_infos = {}
        for i, j in combinations(names, 2):
            kl_ij = sum(q0[s] * bern_kl(theta[i][s], theta[j][s]) for s in subfields)
            kl_ji = sum(q0[s] * bern_kl(theta[j][s], theta[i][s]) for s in subfields)
            kl = (kl_ij + kl_ji) / 2  # 对称化
            pair_infos[(i, j)] = kl
            # 最弱维度
            weakest = min(bern_kl(theta[i][s], theta[j][s]) for s in subfields)
            print(f"  {i} vs {j:<10}{kl:>12.4f}{weakest:>14.4f}")

        i_min = min(pair_infos.values())
        print(f"\n  最坏情况信息量 I_min(q0) = {i_min:.4f}")
        if i_min < 0.01:
            print("  ⚠️ 存在被动观测几乎无法区分的 checkpoint 对 → 具备主动观测的潜在价值")
        else:
            print("  ✅ 被动观测足以区分所有 checkpoint 对 → 主动观测价值有限")


if __name__ == "__main__":
    main()
