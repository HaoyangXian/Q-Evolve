#!/usr/bin/env python3
"""
Branching 结果分析：同位置下 V/Qmin 与后续成功率的关联。

输入：
  prefix_scores_*.jsonl : (qid, traj_idx, prefix_steps, V, Qmin)
  cont_*.jsonl          : (qid, traj_idx, prefix_steps, cont_idx, success, ...)

分析：
  1. 聚合每个前缀的 success_rate = mean(success)
  2. 同 prefix_steps 下，计算 Spearman/Pearson(V, success_rate), (Qmin, success_rate)
  3. 控制 prefix token 长度的偏相关（若可用）
  4. 校准：按 V/Qmin 分桶，看实际成功率
"""
import argparse
import json
from pathlib import Path
import numpy as np

def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="prefix_scores jsonl")
    ap.add_argument("--cont", nargs="+", required=True, help="cont 结果 jsonl")
    args = ap.parse_args()

    # 读取前缀打分
    scores = {}
    for sp in args.scores:
        for r in read_jsonl(sp):
            scores[(r["qid"], r["traj_idx"], r["prefix_steps"])] = r

    # 读取 continuation，聚合成功率
    succ = {}  # (qid, traj_idx, prefix_steps) -> [success bools]
    for cp in args.cont:
        for r in read_jsonl(cp):
            key = (r["qid"], r["traj_idx"], r["prefix_steps"])
            succ.setdefault(key, []).append(1.0 if r.get("success") else 0.0)

    # 合并
    rows = []
    for key, s in scores.items():
        if key not in succ:
            continue
        rate = np.mean(succ[key])
        n = len(succ[key])
        rows.append({
            "qid": key[0], "traj_idx": key[1], "prefix_steps": key[2],
            "V": s["V"], "Qmin": s["Qmin"],
            "success_rate": rate, "n_cont": n,
        })

    if not rows:
        print("[analyze] 无匹配的前缀打分与 continuation，可能 stage_cont 尚未完成")
        return

    print("=" * 72)
    print(f"Branching 分析：共 {len(rows)} 个 (前缀, 成功率) 观测")
    print("=" * 72)

    from scipy.stats import spearmanr, pearsonr

    for ps in sorted(set(r["prefix_steps"] for r in rows)):
        sub = [r for r in rows if r["prefix_steps"] == ps]
        if len(sub) < 5:
            continue
        V = [r["V"] for r in sub]
        Q = [r["Qmin"] for r in sub]
        rate = [r["success_rate"] for r in sub]
        rho_v, p_v = spearmanr(V, rate)
        rho_q, p_q = spearmanr(Q, rate)
        pr_v = pearsonr(V, rate)[0]
        pr_q = pearsonr(Q, rate)[0]
        print(f"\n位置 prefix_steps={ps} (n={len(sub)}):")
        print(f"  V     Spearman={rho_v:+.3f} (p={p_v:.3f})  Pearson={pr_v:+.3f}")
        print(f"  Qmin  Spearman={rho_q:+.3f} (p={p_q:.3f})  Pearson={pr_q:+.3f}")

        # 校准：按 V 分桶看成功率
        if len(sub) >= 10:
            order = np.argsort(V)
            lo = [rate[i] for i in order[:len(order)//2]]
            hi = [rate[i] for i in order[len(order)//2:]]
            print(f"  V 低半桶成功率={np.mean(lo):.3f}  V 高半桶成功率={np.mean(hi):.3f}")
            order_q = np.argsort(Q)
            lo_q = [rate[i] for i in order_q[:len(order_q)//2]]
            hi_q = [rate[i] for i in order_q[len(order_q)//2:]]
            print(f"  Qmin 低半桶成功率={np.mean(lo_q):.3f}  Qmin 高半桶成功率={np.mean(hi_q):.3f}")

    # 汇总所有位置
    print("\n" + "=" * 72)
    print("汇总（跨位置）")
    print("=" * 72)
    V = [r["V"] for r in rows]; Q = [r["Qmin"] for r in rows]; rate = [r["success_rate"] for r in rows]
    ps = [r["prefix_steps"] for r in rows]
    rho_v, p_v = spearmanr(V, rate)
    rho_q, p_q = spearmanr(Q, rate)
    print(f"V     Spearman={rho_v:+.3f} (p={p_v:.3f})")
    print(f"Qmin  Spearman={rho_q:+.3f} (p={p_q:.3f})")
    # 控制位置的部分相关
    def partial(x, y, z):
        z = np.asarray(z, float)
        for v in (x, y):
            pass
        def resid(a, c):
            c = np.asarray(c, float)
            coef = np.polyfit(c, a, 1)
            return np.asarray(a, float) - (coef[0]*c + coef[1])
        return pearsonr(resid(x, z), resid(y, z))[0]
    pc_v = partial(V, rate, ps)
    pc_q = partial(Q, rate, ps)
    print(f"控制位置后: V 偏相关={pc_v:+.3f}, Qmin 偏相关={pc_q:+.3f}")

if __name__ == "__main__":
    main()
