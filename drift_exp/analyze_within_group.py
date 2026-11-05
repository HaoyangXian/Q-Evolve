#!/usr/bin/env python3
"""
补充分析：组内 concordance（同题同位置内，V/Qmin 更高的轨迹是否成功率更高）。

这是 Branching 实验最核心的检验——消除题目间混淆，只在同一道题、同一
推理位置内比较 3 条独立轨迹：V 排序是否与成功率排序一致。

concordance 定义：对每组 3 条轨迹的两两对 (i,j)，
  若 sign(V_i - V_j) == sign(rate_i - rate_j) 记 1，
  若方向相反记 0，若 V 或 rate 相等（tie）记 0.5。
随机水平 = 0.5。
"""
import json
from pathlib import Path
from collections import defaultdict

OUT = Path("/data3/xhy/codes/work/paper_repos/Q-Evolve/drift_exp/branch_pilot")

def read(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

# 前缀打分
scores = {}
for sp in [OUT/"prefix_scores_shard0.jsonl", OUT/"prefix_scores_shard1.jsonl"]:
    for r in read(sp):
        scores[(r["qid"], r["traj_idx"], r["prefix_steps"])] = r

# continuation 成功率
succ = defaultdict(list)
for cp in [OUT/"cont_shard0.jsonl", OUT/"cont_shard1.jsonl"]:
    for r in read(cp):
        succ[(r["qid"], r["traj_idx"], r["prefix_steps"])].append(1.0 if r.get("success") else 0.0)

# 按 (qid, prefix_steps) 分组，组内有 3 条轨迹的才能做组内比较
groups = defaultdict(dict)
for key, s in scores.items():
    if key not in succ:
        continue
    rate = sum(succ[key]) / len(succ[key])
    groups[(key[0], key[2])][key[1]] = {"V": s["V"], "Qmin": s["Qmin"], "rate": rate}

def concordance(vals, rates):
    """3 条轨迹两两比较的 concordance，处理 ties。"""
    n = len(vals)
    acc = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            dv = vals[i] - vals[j]
            dr = rates[i] - rates[j]
            if dv == 0 or dr == 0:
                acc += 0.5  # tie
            elif (dv > 0) == (dr > 0):
                acc += 1.0
            else:
                acc += 0.0
            cnt += 1
    return acc / cnt if cnt else 0.5

# 只保留组内有 >=3 条轨迹、且成功率有差异（有区分度）的组
for metric in ["V", "Qmin"]:
    concs_all = []
    concs_diff = []  # 只算成功率有差异的组（rate 不全相同）
    for (qid, ps), trajs in groups.items():
        if len(trajs) < 3:
            continue
        vals = [trajs[t][metric] for t in sorted(trajs)]
        rates = [trajs[t]["rate"] for t in sorted(trajs)]
        c = concordance(vals, rates)
        concs_all.append(c)
        if len(set(rates)) > 1:  # 成功率有区分度
            concs_diff.append(c)

    import statistics
    print(f"=== {metric} 组内 concordance ===")
    print(f"  有效组数(>=3轨迹): {len(concs_all)}")
    print(f"  有区分度组数(rate不全同): {len(concs_diff)}")
    if concs_all:
        print(f"  全部组 concordance 均值 = {statistics.mean(concs_all):.3f}  (随机=0.500)")
    if concs_diff:
        print(f"  有区分度组 concordance 均值 = {statistics.mean(concs_diff):.3f}  (随机=0.500)")
        # 单样本符号检验近似：偏离 0.5 是否显著
        import math
        m = statistics.mean(concs_diff)
        se = statistics.stdev(concs_diff) / math.sqrt(len(concs_diff)) if len(concs_diff) > 1 else 0
        print(f"  有区分度组 std={statistics.stdev(concs_diff):.3f} se={se:.3f}")
    print()

# 分位置报告
print("=== 分位置 组内 concordance（有区分度组）===")
for ps in sorted(set(k[1] for k in groups)):
    for metric in ["V", "Qmin"]:
        concs = []
        for (qid, p), trajs in groups.items():
            if p != ps or len(trajs) < 3:
                continue
            vals = [trajs[t][metric] for t in sorted(trajs)]
            rates = [trajs[t]["rate"] for t in sorted(trajs)]
            if len(set(rates)) > 1:
                concs.append(concordance(vals, rates))
        if concs:
            import statistics
            print(f"  ps={ps} {metric}: n={len(concs)}  concordance={statistics.mean(concs):.3f}")
