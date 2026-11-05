#!/usr/bin/env python3
"""
P0: Position-only Baseline
在同一 189 题上，比较 final_V 与「位置/长度/终止」简单特征的检测能力，
并用 question-level 分组交叉验证评估 final_V 的增量。
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---------- 确定性映射 ----------
det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2],
                    'ep_len': int(p[3]), 'n_tokens': int(p[4]), 'max_next_end': int(p[5])})
A_by_tid = {d['traj_id']: d for d in det}

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
solB = load(ROOT / 'drift_exp/solver_B_ms64.jsonl')
solC = load(ROOT / 'drift_exp/solver_C_ms32.jsonl')

def get_adv(nm):
    return load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{nm}/adv_token_pred-gamma0.95-len32768-{nm}.jsonl')

# 每档：qid -> 特征 dict
def build(g, solver, adv, use_det):
    out = {}
    for r in adv:
        if use_det:
            d = A_by_tid.get(r['traj_id'])
            if d is None: continue
            row = d['solver_row']
            n_tokens = d['n_tokens']
        else:
            if r['traj_id'] >= len(solver): continue
            row = r['traj_id']
            n_tokens = None
        s = solver[row]
        V = r.get('V') or []
        if not V: continue
        out[s['id']] = {
            'ep_len': r.get('ep_len'),
            'final_V': V[-1],
            'mean_V': sum(V)/len(V),
            'n_tokens': n_tokens,
            'success': bool((s.get('meta') or {}).get('success')),
        }
    return out

D = {
    'A': build('A', solA, get_adv('drift_A_ms512'), True),
    'B': build('B', solB, get_adv('drift_B_ms64'), False),
    'C': build('C', solC, get_adv('drift_C_ms32'), False),
}
common = sorted(set(D['A']) & set(D['B']) & set(D['C']))
print('共同题目:', len(common))

# ---------- 构造样本：每道题 3 个观测（A/B/C），label = 是否退化 ----------
# 两个任务：A vs B, A vs C, 以及 A vs B+C
def make_dataset(neg_groups, pos_groups):
    """neg=正常(A), pos=退化(B/C)。返回 list of (features, label, qid)"""
    rows = []
    for g in neg_groups:
        for q in common:
            rows.append((D[g][q], 0, q))
    for g in pos_groups:
        for q in common:
            rows.append((D[g][q], 1, q))
    return rows

# ---------- 单变量 AUC（ties-aware）----------
def auc_ties(pos, neg):
    if len(pos)==0 or len(neg)==0: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:,None] > neg[None,:]).sum()
    eq = (pos[:,None] == neg[None,:]).sum()
    return (gt + 0.5*eq)/(len(pos)*len(neg))

def uni_auc(ds, key, direction):
    pos = [direction * r[0][key] for r in ds if r[1]==1]
    neg = [direction * r[0][key] for r in ds if r[1]==0]
    return auc_ties(pos, neg)

print()
print('=' * 84)
print('P0-A: 单变量 AUC（方向统一为「越高越退化」）')
print('=' * 84)

feats = [
    ('ep_len (step index)', 'ep_len', -1),
    ('final_V', 'final_V', -1),
    ('mean_V', 'mean_V', -1),
    ('n_tokens', 'n_tokens', -1),
]
for label, key, direction in feats:
    for neg, pos, lab in [('A', ('B','C'), 'A-vs-BC'), ('A', ('B',), 'A-vs-B'), ('A', ('C',), 'A-vs-C')]:
        ds = make_dataset([neg], list(pos))
        a = uni_auc(ds, key, direction)
        print('  %-20s  %-10s  AUC = %.3f' % (label, lab, a))
    print()

# ---------- question-level bootstrap CI ----------
print('=' * 84)
print('P0-B: question-level bootstrap 95% CI（A-vs-BC）')
print('=' * 84)
rng = np.random.default_rng(42)
B = 2000
qs = np.array(common)

def boot_auc(key, direction, neg_g, pos_gs):
    ests = []
    for _ in range(B):
        idx = rng.integers(0, len(qs), len(qs))
        sub = qs[idx]
        pos = []; neg = []
        for q in sub:
            neg.append(direction * D[neg_g][q][key])
        for q in sub:
            for g in pos_gs:
                pos.append(direction * D[g][q][key])
        v = auc_ties(pos, neg)
        if not np.isnan(v): ests.append(v)
    return np.percentile(ests, [2.5, 97.5])

for label, key, direction in feats:
    if key == 'n_tokens' and any(D[g][q]['n_tokens'] is None for g in 'ABC' for q in common):
        # n_tokens 仅 A 档有，跳过 bootstrap（只用 A 档单变量）
        continue
    lo, hi = boot_auc(key, direction, 'A', ('B','C'))
    a = uni_auc(make_dataset(['A'], ('B','C')), key, direction)
    print('  %-20s AUC=%.3f  [%.3f, %.3f]' % (label, a, lo, hi))

# ---------- 逻辑回归增量分析（question-level K-fold CV）----------
print()
print('=' * 84)
print('P0-C: 逻辑回归增量（question-level 5-fold CV）')
print('=' * 84)
print('任务：A vs B+C 二分类，比较不同特征集的 CV-AUC / Brier / log-loss')
print()

# 用简单 sigmoid + 梯度下降，避免依赖 sklearn
def logistic_cv(feat_cols, folds=5, seed=42):
    """feat_cols: list of key。返回 CV 预测概率（与真实 label 对齐）"""
    ds = make_dataset(['A'], ('B','C'))
    rng = np.random.default_rng(seed)
    # 按 question 分组
    q_by_id = {}
    for r in ds:
        q_by_id.setdefault(r[2], []).append(r)
    qids = list(q_by_id.keys())
    rng.shuffle(qids)
    preds = np.full(len(ds), np.nan)
    y_true = np.array([r[1] for r in ds])
    # 建立行索引映射
    for fold in range(folds):
        test_q = set(qids[fold::folds])
        train_idx = [i for i, r in enumerate(ds) if r[2] not in test_q]
        test_idx = [i for i, r in enumerate(ds) if r[2] in test_q]
        Xtr = np.array([[r[0][c] for c in feat_cols] for i, r in enumerate(ds) if i in train_idx], float)
        ytr = np.array([ds[i][1] for i in train_idx], float)
        Xte = np.array([[r[0][c] for c in feat_cols] for i, r in enumerate(ds) if i in test_idx], float)
        # 标准化
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu)/sd; Xte = (Xte - mu)/sd
        # 逻辑回归梯度下降
        w = np.zeros(Xtr.shape[1]); b = 0.0
        lr = 0.1
        for _ in range(2000):
            z = Xtr @ w + b
            p = 1/(1+np.exp(-z))
            gw = Xtr.T @ (p - ytr)/len(ytr) + 0.01*w
            gb = (p - ytr).mean()
            w -= lr*gw; b -= lr*gb
        z = Xte @ w + b
        preds[test_idx] = 1/(1+np.exp(-z))
    return preds, y_true

def report(name, preds, y):
    # AUC
    pos = preds[y==1]; neg = preds[y==0]
    a = auc_ties(pos.tolist(), neg.tolist())
    # Brier
    brier = np.mean((preds - y)**2)
    # log-loss
    eps = 1e-12
    ll = -np.mean(y*np.log(preds+eps) + (1-y)*np.log(1-preds+eps))
    print('  %-32s AUC=%.3f  Brier=%.4f  LogLoss=%.4f' % (name, a, brier, ll))
    return a

print('（特征均标准化；position 用 ep_len，含 -ep_len 方向 = 更短更退化）')
# 特征集
sets = {
    'position (ep_len)': ['ep_len'],
    'final_V': ['final_V'],
    'position + final_V': ['ep_len', 'final_V'],
}
res = {}
for name, cols in sets.items():
    preds, y = logistic_cv(cols)
    res[name] = report(name, preds, y)

print()
print('==> 增量判断：')
p1, y1 = logistic_cv(['ep_len'])
p2, y2 = logistic_cv(['ep_len', 'final_V'])
a1 = auc_ties(p1[y1==1].tolist(), p1[y1==0].tolist())
a2 = auc_ties(p2[y2==1].tolist(), p2[y2==0].tolist())
print('  AUC(position+final_V) - AUC(position) = %+.3f' % (a2 - a1))

# 用 bootstrap 估计增量的 CI
rng = np.random.default_rng(7)
ds = make_dataset(['A'], ('B','C'))
q_by_id = {}
for i, r in enumerate(ds):
    q_by_id.setdefault(r[2], []).append(i)
qids = list(q_by_id.keys())
diffs = []
for _ in range(1000):
    idx = rng.integers(0, len(qids), len(qids))
    # 简化：用重采样后的 question 集合重算单变量 AUC（不重跑逻辑回归，太慢）
    sub_q = set(qids[i] for i in idx)
    # 单变量 AUC
    def ua(key, direction):
        pos = []; neg = []
        for i, r in enumerate(ds):
            if r[2] not in sub_q: continue
            if r[1]==1: pos.append(direction*r[0][key])
            else: neg.append(direction*r[0][key])
        return auc_ties(pos, neg)
    diffs.append(ua('final_V', -1) - ua('ep_len', -1))
diffs = np.array(diffs)
print('  ΔAUC(final_V - position) 单变量 bootstrap 95%%CI = [%.3f, %.3f]' % tuple(np.percentile(diffs, [2.5, 97.5])))
