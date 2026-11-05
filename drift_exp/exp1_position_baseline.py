#!/usr/bin/env python3
"""
Experiment 1: Position-only Baseline (P0, 无需 GPU 训练)
判断 final_V 相对「位置/长度/终止」等朴素信号是否有增量。

预测目标：A(正常) vs B+C(截断)  —— 注意这是预测"截断档位"，不是潜在能力
评估：question-level GroupKFold(5)，完全按 qid 分组，避免同题泄漏
指标：AUC / log-loss / Brier
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'QEvolve'))

from transformers import AutoTokenizer
from eval_agent.data_utils import get_chat_template

MAX_LENGTH = 32768
MAX_STEPS = {'A': 512, 'B': 64, 'C': 32}
MODEL_NAME = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/sft_success')

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---------- 1. 读确定性映射 ----------
det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2],
                    'ep_len': int(p[3]), 'n_tokens': int(p[4])})
A_row = {d['traj_id']: d['solver_row'] for d in det}
A_ntok = {int(d['solver_row']): d['n_tokens'] for d in det}
A_ep = {int(d['solver_row']): d['ep_len'] for d in det}

# ---------- 2. 计算 B/C 的 token 数 ----------
print('[tokenize] 计算 B/C 的 token 长度...', file=sys.stderr)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token
tmpl = get_chat_template(MODEL_NAME)
roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

def n_tokens_of(item, max_steps):
    conv = item.get('conversations')
    if not conv:
        return None
    n_msgs = len(conv)
    step_idxs = [i for i in range(2, n_msgs, 2) if i + 1 < n_msgs][:max_steps]
    if not step_idxs:
        return None
    msgs = []
    for i, sent in enumerate(conv):
        try:
            role = roles[sent["from"]]
        except KeyError:
            return None
        msgs.append({"role": role, "content": sent["value"]})
    acc = 0
    offsets = []
    for m in msgs:
        pre, suf = tmpl.role_prefix_and_suffix.get(m['role'], ('', ''))
        ids = tok(pre + m['content'] + suf, add_special_tokens=False).input_ids
        acc += len(ids)
        offsets.append(acc)
    next_end = [(offsets[i+2]-1) if (i+2 < n_msgs) else (offsets[i+1]-1) for i in step_idxs]
    return next_end[-1] + 1

sols = {}
for k, fn in [('A','solver_A_ms512'), ('B','solver_B_ms64'), ('C','solver_C_ms32')]:
    sols[k] = load(ROOT / f'drift_exp/{fn}.jsonl')

ntok = {'B': {}, 'C': {}}
for k in ['B', 'C']:
    for idx, item in enumerate(sols[k]):
        ntok[k][idx] = n_tokens_of(item, MAX_STEPS[k])
print('[tokenize] 完成', file=sys.stderr)

# ---------- 3. 组装样本 ----------
def agg(rec):
    V = rec.get('V') or []
    return {'final_V': V[-1], 'mean_V': sum(V)/len(V)} if V else None

advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
advB = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_B_ms64/adv_token_pred-gamma0.95-len32768-drift_B_ms64.jsonl')
advC = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_C_ms32/adv_token_pred-gamma0.95-len32768-drift_C_ms32.jsonl')

samples = []   # (qid, group, label, feats)
for r in advA:
    a = agg(r)
    if not a: continue
    row = A_row.get(r['traj_id'])
    if row is None: continue
    q = sols['A'][row]['id']
    ep = A_ep[row]; nt = A_ntok[row]
    samples.append(dict(qid=q, group='A', label=0, ep=ep, ntok=nt,
                        trunc=1 if ep >= MAX_STEPS['A'] else 0, final_V=a['final_V']))
for k, adv in [('B', advB), ('C', advC)]:
    for r in adv:
        a = agg(r)
        if not a: continue
        idx = r['traj_id']
        if idx >= len(sols[k]): continue
        q = sols[k][idx]['id']
        ep = r['ep_len']; nt = ntok[k].get(idx)
        if nt is None: continue
        samples.append(dict(qid=q, group=k, label=1, ep=ep, ntok=nt,
                            trunc=1 if ep >= MAX_STEPS[k] else 0, final_V=a['final_V']))

print('样本数:', len(samples), ' A=%d B+C=%d' % (
    sum(1 for s in samples if s['group']=='A'),
    sum(1 for s in samples if s['group']!='A')))

# ---------- 4. AUC（配对/池化）----------
def auc_ties(pos, neg):
    if not pos or not neg: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5*eq) / (len(pos)*len(neg))

print()
print('=' * 78)
print('单特征基线 AUC（预测 A vs B+C 截断档位）')
print('=' * 78)
pos = [s for s in samples if s['label'] == 1]
neg = [s for s in samples if s['label'] == 0]
baselines = {
    'ep (step index)': lambda s: s['ep'],
    'ntok (token len)': lambda s: s['ntok'],
    'trunc (termination)': lambda s: s['trunc'],
    'final_V': lambda s: s['final_V'],
    '-ep': lambda s: -s['ep'],
    '-ntok': lambda s: -s['ntok'],
}
for name, f in baselines.items():
    a = auc_ties([f(s) for s in pos], [f(s) for s in neg])
    print('  %-22s AUC = %.3f' % (name, a))

# ---------- 5. 逻辑回归（自实现）+ GroupKFold ----------
def standardize(X, mu=None, sd=None):
    if mu is None:
        mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd

def fit_logreg(X, y, l2=1.0, lr=0.1, iters=3000):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw; b -= lr * gb
    return w, b

def predict(X, w, b):
    return 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))

def group_kfold(qids, k=5, seed=0):
    uq = np.array(sorted(set(qids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uq)
    folds = np.array_split(uq, k)
    return folds

def eval_feats(feats, name, k=5, seed=0):
    X = np.array([[s[f] for f in feats] for s in samples], float)
    y = np.array([s['label'] for s in samples], float)
    qids = [s['qid'] for s in samples]
    folds = group_kfold(qids, k, seed)
    aucs, lls, briers = [], [], []
    for i in range(k):
        te_q = set(folds[i].tolist())
        tr = np.array([j for j, q in enumerate(qids) if q not in te_q])
        te = np.array([j for j, q in enumerate(qids) if q in te_q])
        if len(tr) == 0 or len(te) == 0: continue
        Xtr, mu, sd = standardize(X[tr])
        Xte = (X[te] - mu) / sd
        w, b = fit_logreg(Xtr, y[tr])
        p = predict(Xte, w, b)
        yt = y[te]
        # AUC
        a = auc_ties(p[yt==1].tolist(), p[yt==0].tolist())
        aucs.append(a)
        eps = 1e-9
        ll = -(yt*np.log(p+eps) + (1-yt)*np.log(1-p+eps)).mean()
        lls.append(ll)
        briers.append(((p - yt)**2).mean())
    return name, np.mean(aucs), np.mean(lls), np.mean(briers)

print()
print('=' * 78)
print('GroupKFold(5, by qid) 交叉验证：模型对比')
print('=' * 78)
configs = [
    (['ep'], 'Step index'),
    (['ntok'], 'Token length'),
    (['trunc'], 'Termination type'),
    (['ep', 'ntok'], 'Step + TokenLen'),
    (['ep', 'ntok', 'trunc'], 'Step + TokenLen + Term'),
    (['ep', 'ntok', 'trunc', 'final_V'], 'Step + TokenLen + Term + final_V'),
    (['final_V'], 'final_V only'),
]
res = {}
print('%-34s %8s %10s %10s' % ('模型', 'AUC', 'log-loss', 'Brier'))
for feats, name in configs:
    r = eval_feats(feats, name)
    res[name] = r
    print('%-34s %8.3f %10.4f %10.4f' % (name, r[1], r[2], r[3]))

base = res['Step + TokenLen + Term']
full = res['Step + TokenLen + Term + final_V']
print()
print('ΔAUC (加 final_V) = %+.4f' % (full[1] - base[1]))
print('Δlog-loss         = %+.4f  (负=更好)' % (full[2] - base[2]))
print('ΔBrier            = %+.4f  (负=更好)' % (full[3] - base[3]))
