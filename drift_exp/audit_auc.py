#!/usr/bin/env python3
# 方向修正后的 AUC 严格审计 + bootstrap CI + 分档分析 + 长度混杂
import json, pathlib, sys
import numpy as np

sys.path.insert(0, '/data3/xhy/codes/work/paper_repos/Q-Evolve')
drift = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/advantage_inference/qa_traj_tokens/wo_assigned_rewards')
names = ['drift_A_ms512', 'drift_B_ms64', 'drift_C_ms32']

def load(name):
    p = drift / name / f'adv_token_pred-gamma0.95-len32768-{name}.jsonl'
    return [json.loads(x) for x in p.open() if x.strip()]

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
        'ep_len': rec.get('ep_len'),
        'is_fail': rec.get('is_fail', False),
    }

def auc(scores, labels):
    pos = sorted(s for s, l in zip(scores, labels) if l == 1)
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return float('nan')
    nn = len(neg)
    s = 0.0
    for p in pos:
        lo, hi = 0, nn
        while lo < hi:
            mid = (lo + hi) // 2
            if neg[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        s += lo
    return s / (len(pos) * nn)

def auc_ci(scores, labels, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    scores = np.array(scores); labels = np.array(labels)
    ests = []
    for _ in range(B):
        idx = rng.integers(0, len(scores), len(scores))
        ests.append(auc(scores[idx].tolist(), labels[idx].tolist()))
    return np.percentile(ests, [2.5, 97.5])

# 收集三档
rows_by_g = []
for name in names:
    g = []
    for rec in load(name):
        a = agg(rec)
        if a:
            g.append(a)
    rows_by_g.append(g)
    print(f'[load] {name}: n={len(g)}', file=sys.stderr)

keys = ['mean_V', 'final_V', 'mean_Qmin', 'mean_A']

print('=' * 70)
print('1) 方向修正后的 AUC（正类=Degraded(1)，Normal(0)；A vs 合并BC）')
print('=' * 70)
print(f'{"指标":<12}{"原始AUC":>10}{"反向AUC":>10}{"95%CI":>22}{"方向":>12}')
for k in keys:
    scores = []; labels = []
    for gi, g in enumerate(rows_by_g):
        for a in g:
            scores.append(a[k]); labels.append(0 if gi == 0 else 1)
    a = auc(scores, labels)
    lo, hi = auc_ci(scores, labels)
    direction = '退化更低' if a < 0.5 else '退化更高'
    print(f'{k:<12}{a:>10.3f}{1-a:>10.3f}   [{lo:.3f},{hi:.3f}]  {direction}')

print()
print('=' * 70)
print('2) 分档两两 AUC（A vs B, A vs C, B vs C），正类=更退化档')
print('=' * 70)
pairs = [(0, 1, 'A-vs-B'), (0, 2, 'A-vs-C'), (1, 2, 'B-vs-C')]
for gi, gj, label in pairs:
    print(f'--- {label} ---')
    for k in keys:
        scores = []; labels_ = []
        for a in rows_by_g[gi]:
            scores.append(a[k]); labels_.append(0)
        for a in rows_by_g[gj]:
            scores.append(a[k]); labels_.append(1)
        a = auc(scores, labels_)
        lo, hi = auc_ci(scores, labels_)
        print(f'  {k:<12} AUC={a:.3f}  反向={1-a:.3f}  95%CI=[{lo:.3f},{hi:.3f}]')

print()
print('=' * 70)
print('3) raw accuracy 分档（正确率）')
print('=' * 70)
from eval_agent.qa_verifier import answers_equal
for name in ['solver_A_ms512', 'solver_B_ms64', 'solver_C_ms32']:
    rows = [json.loads(x) for x in open(f'/data3/xhy/codes/work/paper_repos/Q-Evolve/drift_exp/{name}.jsonl') if x.strip()]
    acc = sum(answers_equal(r.get('pred_answer', ''), r.get('gold_answer', '')) for r in rows) / len(rows)
    print(f'{name}: accuracy={acc:.3f} (n={len(rows)})')

print()
print('=' * 70)
print('4) 长度混杂：ep_len 与各指标的关联 + critic 是否只是在识别长度')
print('=' * 70)
for gi, name in enumerate(names):
    g = rows_by_g[gi]
    lens = [a['ep_len'] for a in g if a['ep_len'] is not None]
    print(f'{name}: ep_len mean={np.mean(lens):.1f} median={np.median(lens):.1f} min={min(lens)} max={max(lens)}')
# ep_len 与 mean_A/mean_V 的相关
print()
print('ep_len 与指标的 Pearson 相关（合并三档）:')
all_a = []
for gi, g in enumerate(rows_by_g):
    for a in g:
        if a['ep_len'] is not None:
            all_a.append((gi, a))
lens = np.array([a['ep_len'] for _, a in all_a])
for k in keys:
    vals = np.array([a[k] for _, a in all_a])
    r = np.corrcoef(lens, vals)[0, 1]
    print(f'  ep_len vs {k:<12} r={r:.3f}')
