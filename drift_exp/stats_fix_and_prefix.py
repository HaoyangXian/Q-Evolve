#!/usr/bin/env python3
"""
统计修正 + D1.1-A 共同前缀一致性
1) 189 道同题上重算 meta.success 的配对 AUC（与 Critic 同题集）
2) ΔAUC 的配对 bootstrap CI（同一 bootstrap 样本内同时算两指标）
3) mean_A 的准确表述（统计可区分 vs 实际判别力）
4) D1.1-A: A/B/C 在共同位置的 V/Q/A 是否完全一致
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---------- 确定性映射（已验证）----------
det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})
A_row = {d['traj_id']: d['solver_row'] for d in det}

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
solB = load(ROOT / 'drift_exp/solver_B_ms64.jsonl')
solC = load(ROOT / 'drift_exp/solver_C_ms32.jsonl')

def get_adv(nm):
    return load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{nm}/adv_token_pred-gamma0.95-len32768-{nm}.jsonl')

advA_raw = get_adv('drift_A_ms512')
advB_raw = get_adv('drift_B_ms64')
advC_raw = get_adv('drift_C_ms32')

def agg(rec):
    V = rec.get('V') or []; Q = rec.get('Qmin') or []; A = rec.get('A') or []
    if not V: return None
    return {'mean_V': sum(V)/len(V), 'final_V': V[-1],
            'mean_Qmin': sum(Q)/len(Q) if Q else float('nan'),
            'mean_A': sum(A)/len(A) if A else float('nan'),
            'ep_len': rec.get('ep_len'), 'V': V, 'Q': Q, 'A': A}

# qid -> {agg, solver_record}
D = {'A': {}, 'B': {}, 'C': {}}
for r in advA_raw:
    a = agg(r)
    if a:
        row = A_row.get(r['traj_id'])
        if row is not None:
            D['A'][solA[row]['id']] = (a, solA[row])
for r in advB_raw:
    a = agg(r)
    if a and r['traj_id'] < len(solB):
        D['B'][solB[r['traj_id']]['id']] = (a, solB[r['traj_id']])
for r in advC_raw:
    a = agg(r)
    if a and r['traj_id'] < len(solC):
        D['C'][solC[r['traj_id']]['id']] = (a, solC[r['traj_id']])

common = sorted(set(D['A']) & set(D['B']) & set(D['C']))
print('三档共同题目:', len(common))

def auc_ties(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

DIR = {'mean_V': -1, 'final_V': -1, 'mean_Qmin': -1, 'mean_A': +1}
KEYS = ['final_V', 'mean_Qmin', 'mean_V', 'mean_A']

# ---------- 189 同题上的 meta.success 配对 AUC ----------
def success_score(g, q):
    _, s = D[g][q]
    return 0.0 if (s.get('meta') or {}).get('success') else 1.0   # 失败率，越高越退化

print()
print('=' * 80)
print('1) 同一 189 题上的 meta.success 配对 AUC（与 Critic 同题集）')
print('=' * 80)
for lo, hi, lab in [('A','B','A-vs-B'), ('A','C','A-vs-C'), ('B','C','B-vs-C')]:
    pos = [success_score(hi, q) for q in common]
    neg = [success_score(lo, q) for q in common]
    print('  %-8s meta.success 失败率 AUC = %.3f' % (lab, auc_ties(pos, neg)))
pos = [success_score(g, q) for g in ('B','C') for q in common]
neg = [success_score('A', q) for q in common]
print('  %-8s meta.success 失败率 AUC = %.3f' % ('A-vs-BC', auc_ties(pos, neg)))

# ---------- ΔAUC 配对 bootstrap ----------
print()
print('=' * 80)
print('2) ΔAUC 的配对 bootstrap 95% CI（同一重采样样本内同时计算）')
print('=' * 80)

def auc_for(k, lo_g, hi_g, qs):
    pos = [DIR[k] * D[g][q][0][k] for g in hi_g for q in qs]
    neg = [DIR[k] * D['A'][q][0][k] for q in qs]
    return auc_ties(pos, neg)

rng = np.random.default_rng(42)
B = 2000
qs = np.array(common)
boot = {k: [] for k in KEYS}
boot_acc = []
for _ in range(B):
    idx = rng.integers(0, len(qs), len(qs))
    sub = qs[idx].tolist()
    for k in KEYS:
        boot[k].append(auc_for(k, 'A', ('B','C'), sub))
    p = [success_score(g, q) for g in ('B','C') for q in sub]
    n = [success_score('A', q) for q in sub]
    boot_acc.append(auc_ties(p, n))

boot_acc = np.array(boot_acc)
print('%-22s %10s %12s %22s' % ('比较', 'ΔAUC', '95%CI', '是否显著(不含0)'))
for k in KEYS:
    arr = np.array(boot[k])
    base_v = auc_for(k, 'A', ('B','C'), common)
    for ref_name, ref_arr, ref_v in [('mean_Qmin', np.array(boot['mean_Qmin']), auc_for('mean_Qmin','A',('B','C'),common)),
                                     ('mean_V', np.array(boot['mean_V']), auc_for('mean_V','A',('B','C'),common)),
                                     ('mean_A', np.array(boot['mean_A']), auc_for('mean_A','A',('B','C'),common)),
                                     ('meta.success', boot_acc, auc_ties([success_score(g,q) for g in ('B','C') for q in common],[success_score('A',q) for q in common]))]:
        if k == ref_name: continue
        d = arr - ref_arr
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = 'YES' if (lo > 0 or hi < 0) else 'no'
        print('%-22s %+10.3f  [%+.3f, %+.3f]   %s' % (f'{k} - {ref_name}', base_v - ref_v, lo, hi, sig))
    print()

# ---------- mean_A 的准确表述 ----------
print('=' * 80)
print('3) 各指标总体 AUC 与 CI（用于区分 statistical vs practical）')
print('=' * 80)
print('%-12s %8s %22s %14s' % ('指标', 'AUC', '95%CI', 'CI 下界>0.5'))
for k in KEYS:
    arr = np.array(boot[k]); v = auc_for(k, 'A', ('B','C'), common)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print('%-12s %8.3f   [%.3f, %.3f]   %s' % (k, v, lo, hi, 'YES' if lo > 0.5 else 'NO'))
lo, hi = np.percentile(boot_acc, [2.5, 97.5])
v = auc_ties([success_score(g,q) for g in ('B','C') for q in common],[success_score('A',q) for q in common])
print('%-12s %8.3f   [%.3f, %.3f]   %s' % ('meta.success', v, lo, hi, 'YES' if lo > 0.5 else 'NO'))

# ---------- D1.1-A 共同前缀一致性 ----------
print()
print('=' * 80)
print('4) D1.1-A 共同前缀一致性：A/B/C 在共同位置的 V/Q/A 是否相同')
print('=' * 80)

def compare_prefix(k_arr, lim):
    """对每道题取前 lim 个 step，比较三档"""
    res = {'A_B': [], 'A_C': [], 'B_C': []}
    for q in common:
        a = D['A'][q][0][k_arr][:lim]
        b = D['B'][q][0][k_arr][:lim]
        c = D['C'][q][0][k_arr][:lim]
        L = min(len(a), len(b), len(c))
        if L == 0: continue
        res['A_B'].append(np.abs(np.array(a[:L]) - np.array(b[:L])).max())
        res['A_C'].append(np.abs(np.array(a[:L]) - np.array(c[:L])).max())
        res['B_C'].append(np.abs(np.array(b[:L]) - np.array(c[:L])).max())
    return res

for k_arr, label in [('V', 'V'), ('Q', 'Qmin'), ('A', 'A')]:
    print(f'--- {label} ---')
    for lim in [8, 16, 32]:
        r = compare_prefix(k_arr, lim)
        line = []
        for pair in ['A_B', 'A_C', 'B_C']:
            arr = np.array(r[pair])
            line.append('%s max=%.2e exact=%.0f%%' % (pair, arr.max(), 100*(arr == 0).mean()))
        print('  前 %-3d 步: %s' % (lim, '  |  '.join(line)))
    # 64 步只对 A-B（C 只有 32）
    r = compare_prefix(k_arr, 64)
    arr = np.array(r['A_B'])
    print('  前 64 步(A-B): max=%.2e  exact=%.0f%%' % (arr.max(), 100*(arr == 0).mean()))
    print()
