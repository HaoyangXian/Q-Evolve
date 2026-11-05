#!/usr/bin/env python3
# 实验A：聚合 critic 输出 + 计算 AUC(critic, drift) vs AUC(raw_accuracy, drift) + 负对照
# 用法（在 $ROOT 下运行，需 PYTHONPATH=$ROOT）:
#   PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
#   $PY bin/qa/calc_drift_auc.py \
#       --critic advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl \
#                advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_B_ms32/adv_token_pred-gamma0.95-len32768-drift_B_ms32.jsonl \
#                advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_C_ms8/adv_token_pred-gamma0.95-len32768-drift_C_ms8.jsonl \
#       --solvers drift_exp/solver_A_ms512.jsonl drift_exp/solver_B_ms32.jsonl drift_exp/solver_C_ms8.jsonl \
#       --base data/qa/olympiad_eval200_seed20261031/base.jsonl
import json, argparse, sys

def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]

def auc(scores, labels):
    """Mann-Whitney U 实现的二分类 AUC，零依赖。labels: 0/1。"""
    pos = sorted(s for s, l in zip(scores, labels) if l == 1)
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float('nan')
    n_neg = len(neg)
    s = 0.0
    for p in pos:
        lo, hi = 0, n_neg
        while lo < hi:
            mid = (lo + hi) // 2
            if neg[mid] < p: lo = mid + 1
            else: hi = mid
        s += lo
    return s / (len(pos) * n_neg)

def agg(rec):
    V = rec.get('V') or []
    Q = rec.get('Qmin') or []
    A = rec.get('A') or []
    if not V:
        return None
    return {
        'mean_V': sum(V) / len(V),
        'final_V': V[-1],
        'mean_Qmin': sum(Q) / len(Q) if Q else float('nan'),
        'mean_A': sum(A) / len(A) if A else float('nan'),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', nargs='+', required=True)
    ap.add_argument('--solvers', nargs='+', required=True)
    ap.add_argument('--base', default=None)
    args = ap.parse_args()

    assert len(args.critic) == len(args.solvers), "critic 与 solvers 组数不一致"

    try:
        from eval_agent.qa_verifier import answers_equal
    except Exception as e:
        print(f"[fatal] 无法 import answers_equal: {e}", file=sys.stderr)
        sys.exit(3)

    # ---- 对齐：优先按 id join（critic 记录若无 id 则回退行序）----
    def align(crit_path, sol_path):
        crits = load_jsonl(crit_path)
        sols = load_jsonl(sol_path)
        if crits and sols and 'id' in crits[0] and 'id' in sols[0]:
            cid = {r['id']: r for r in crits}
            return [(cid.get(s['id']), s) for s in sols if s.get('id') in cid]
        return list(zip(crits, sols))  # 行序回退

    print('=== 主实验：Normal(0) vs Degraded(1) 的区分度 ===')
    keys = ['mean_V', 'final_V', 'mean_Qmin', 'mean_A']
    for k in keys:
        scores, labels = [], []
        for gi in range(len(args.critic)):
            for rec in load_jsonl(args.critic[gi]):
                a = agg(rec)
                if a is not None:
                    scores.append(a[k])
                    labels.append(0 if gi == 0 else 1)
        print(f'AUC({k:>10}, drift) = {auc(scores, labels):.3f}   (n={len(scores)})')

    # ---- raw accuracy baseline ----
    acc_scores, acc_labels = [], []
    for gi, sol_path in enumerate(args.solvers):
        for r in load_jsonl(sol_path):
            ok = answers_equal(r.get('pred_answer', ''), r.get('gold_answer', ''))
            acc_scores.append(1.0 if ok else 0.0)
            acc_labels.append(0 if gi == 0 else 1)
    print(f'AUC(raw_accuracy, drift) = {auc(acc_scores, acc_labels):.3f}   (n={len(acc_scores)})')

    # ---- 负对照：同一 Solver-A，按 base 正确性分易/难题，看 critic mean_V 是否被难度扰动 ----
    if args.base:
        print('\n=== 负对照：Solver-A 在易/难题上的 critic 分数（难度混杂检验）===')
        base_rows = load_jsonl(args.base)
        id2ok = {}
        for b in base_rows:
            bid = b.get('id')
            if bid is None:
                continue
            id2ok[bid] = answers_equal(b.get('pred_answer', ''), b.get('gold_answer', ''))
        pairs = align(args.critic[0], args.solvers[0])
        easy_v, hard_v = [], []
        for crit, sol in pairs:
            if crit is None or sol is None:
                continue
            a = agg(crit)
            if a is None:
                continue
            base_ok = id2ok.get(sol.get('id'))
            if base_ok is None:
                continue
            (easy_v if base_ok else hard_v).append(a['mean_V'])
        if easy_v and hard_v:
            me = sum(easy_v) / len(easy_v)
            mh = sum(hard_v) / len(hard_v)
            print(f'易题(base做对) mean_V = {me:.4f} (n={len(easy_v)})')
            print(f'难题(base做错) mean_V = {mh:.4f} (n={len(hard_v)})')
            print(f'难度造成的分数差 = {abs(me - mh):.4f}  (越小越不易混淆难度)')
        else:
            print(f'负对照数据不足：easy_n={len(easy_v)} hard_n={len(hard_v)}，请核对 base.jsonl 的 id 对齐')

if __name__ == '__main__':
    main()
