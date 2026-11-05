#!/usr/bin/env python3
# P0 修正版：逻辑回归增量分析（question-level 5-fold CV）
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})
A_by_tid = {d['traj_id']: d for d in det}

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
solB = load(ROOT / 'drift_exp/solver_B_ms64.jsonl')
solC = load(ROOT / 'drift_exp/solver_C_ms32.jsonl')

def get_adv(nm):
    return load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{nm}/adv_token_pred-gamma0.95-len32768-{nm}.jsonl')

def build(g, solver, adv, use_det):
    out = {}
    for r in adv:
        if use_det:
            d = A_by_tid.get(r['traj_id'])
            if d is None: continue
            row = d['solver_row']
        else:
            if r['traj_id'] >= len(solver): continue
            row = r['traj_id']
        s = solver[row]
        V = r.get('V') or []
        if not V: continue
        Q = r.get('Qmin') or []
        A = r.get('A') or []
        out[s['id']] = {
            'ep_len': r.get('ep_len'),
            'final_V': V[-1],
            'mean_V': sum(V)/len(V),
            'final_Qmin': Q[-1] if Q else float('nan'),
            'final_A': A[-1] if A else float('nan'),
            'success': bool((s.get('meta') or {}).get('success')),
            'pred_eq': bool(answers_equal(s.get('pred_answer',''), s.get('gold_answer',''))),
        }
    return out

D = {
    'A': build('A', solA, get_adv('drift_A_ms512'), True),
    'B': build('B', solB, get_adv('drift_B_ms64'), False),
    'C': build('C', solC, get_adv('drift_C_ms32'), False),
}
common = sorted(set(D['A']) & set(D['B']) & set(D['C']))

def auc_ties(pos, neg):
    if len(pos)==0 or len(neg)==0: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:,None] > neg[None,:]).sum(); eq = (pos[:,None] == neg[None,:]).sum()
    return (gt + 0.5*eq)/(len(pos)*len(neg))

def make_dataset(neg_groups, pos_groups):
    rows = []
    for g in neg_groups:
        for q in common: rows.append((D[g][q], 0, q))
    for g in pos_groups:
        for q in common: rows.append((D[g][q], 1, q))
    return rows

# 方向：越高越退化。position/ep_len 越小越退化 -> 取负；V/Qmin/A 越小越退化 -> 取负
DIR = {'ep_len': -1, 'final_V': -1, 'mean_V': -1, 'final_Qmin': -1, 'final_A': +1}

def logistic_cv(feat_cols, folds=5, seed=42):
    ds = make_dataset(['A'], ('B','C'))
    rng = np.random.default_rng(seed)
    q_by_id = {}
    for r in ds: q_by_id.setdefault(r[2], []).append(r)
    qids = list(q_by_id.keys()); rng.shuffle(qids)
    preds = np.full(len(ds), np.nan)
    y_true = np.array([r[1] for r in ds])
    for fold in range(folds):
        test_q = set(qids[fold::folds])
        train_idx = [i for i,r in enumerate(ds) if r[2] not in test_q]
        test_idx = [i for i,r in enumerate(ds) if r[2] in test_q]
        Xtr = np.array([[DIR[c]*r[0][c] for c in feat_cols] for i,r in enumerate(ds) if i in train_idx], float)
        ytr = np.array([ds[i][1] for i in train_idx], float)
        Xte = np.array([[DIR[c]*r[0][c] for c in feat_cols] for i,r in enumerate(ds) if i in test_idx], float)
        mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
        Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
        w = np.zeros(Xtr.shape[1]); b = 0.0; lr = 0.1
        for _ in range(3000):
            z = Xtr@w + b; p = 1/(1+np.exp(-z))
            gw = Xtr.T@(p-ytr)/len(ytr) + 0.01*w; gb = (p-ytr).mean()
            w -= lr*gw; b -= lr*gb
        z = Xte@w + b
        preds[test_idx] = 1/(1+np.exp(-z))
    return preds, y_true

def report(name, preds, y):
    pos = preds[y==1]; neg = preds[y==0]
    a = auc_ties(pos.tolist(), neg.tolist())
    brier = np.mean((preds-y)**2)
    eps = 1e-12
    ll = -np.mean(y*np.log(preds+eps)+(1-y)*np.log(1-preds+eps))
    print('  %-34s AUC=%.3f  Brier=%.4f  LogLoss=%.4f' % (name, a, brier, ll))
    return a, brier, ll

print('=' * 84)
print('P0-C: 逻辑回归增量（question-level 5-fold CV，任务 A vs B+C）')
print('=' * 84)
configs = {
    'position (ep_len)': ['ep_len'],
    'final_V': ['final_V'],
    'mean_V': ['mean_V'],
    'position + final_V': ['ep_len', 'final_V'],
    'position + mean_V': ['ep_len', 'mean_V'],
    'position + final_Qmin': ['ep_len', 'final_Qmin'],
    'position + final_V + final_A': ['ep_len', 'final_V', 'final_A'],
}
res = {}
for name, cols in configs.items():
    preds, y = logistic_cv(cols)
    res[name] = report(name, preds, y)

print()
print('==> 增量（相对 position-only）：')
base_a = res['position (ep_len)'][0]
for name in ['final_V', 'position + final_V', 'position + mean_V', 'position + final_Qmin']:
    a = res[name][0]
    print('  AUC(%s) - AUC(position) = %+.3f' % (name, a - base_a))

# ---------- 用 single question leave-one-out 估计 ΔAUC(final_V - position) 的分布 ----------
print()
print('=' * 84)
print('P0-D: ΔAUC(final_V vs position) 的 question-level bootstrap CI')
print('=' * 84)
rng = np.random.default_rng(7)
ds = make_dataset(['A'], ('B','C'))
def ua(key):
    pos = [DIR[key]*r[0][key] for r in ds if r[1]==1]
    neg = [DIR[key]*r[0][key] for r in ds if r[1]==0]
    return auc_ties(pos, neg)
base_fv = ua('final_V'); base_ep = ua('ep_len')
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(common), len(common))
    sub = set(common[i] for i in idx)
    pos_fv=[]; neg_fv=[]; pos_ep=[]; neg_ep=[]
    for r in ds:
        if r[2] not in sub: continue
        if r[1]==1:
            pos_fv.append(-r[0]['final_V']); pos_ep.append(-r[0]['ep_len'])
        else:
            neg_fv.append(-r[0]['final_V']); neg_ep.append(-r[0]['ep_len'])
    a_fv = auc_ties(pos_fv, neg_fv)
    a_ep = auc_ties(pos_ep, neg_ep)
    diffs.append(a_fv - a_ep)
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print('final_V 单变量 AUC = %.3f' % base_fv)
print('position 单变量 AUC = %.3f' % base_ep)
print('ΔAUC(final_V - position) = %+.3f  [%.3f, %.3f]' % (base_fv-base_ep, lo, hi))
print('增量显著(CI 不含 0):', 'YES' if (lo>0 or hi<0) else 'no')

# ---------- 终止类型分析 ----------
print()
print('=' * 84)
print('P0-E: 终止类型分布（budget 截断 vs 自然结束）')
print('=' * 84)
for g, maxs in [('A', 512), ('B', 64), ('C', 32)]:
    eps = [D[g][q]['ep_len'] for q in common]
    n_cut = sum(1 for e in eps if e >= maxs)
    n_nat = len(eps) - n_cut
    print('  %s: 自然结束(ep<%d)=%d  预算截断(ep=%d)=%d' % (g, maxs, n_nat, maxs, n_cut))
print()
print('  → B/C 大量预算截断，ep_len 直接等于 max_steps，位置信息几乎完全泄露档位。')
