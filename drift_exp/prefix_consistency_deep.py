#!/usr/bin/env python3
"""
D1.1-A 深化：共同前缀一致性的定量刻画
- MAD / MaxAD / Pearson / Spearman
- 差异是否集中在少数题目
- 与轨迹长度、batch 的关系
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

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

D = {'A': {}, 'B': {}, 'C': {}}
for r in get_adv('drift_A_ms512'):
    row = A_row.get(r['traj_id'])
    if row is not None:
        D['A'][solA[row]['id']] = r
for r in get_adv('drift_B_ms64'):
    if r['traj_id'] < len(solB):
        D['B'][solB[r['traj_id']]['id']] = r
for r in get_adv('drift_C_ms32'):
    if r['traj_id'] < len(solC):
        D['C'][solC[r['traj_id']]['id']] = r

common = sorted(set(D['A']) & set(D['B']) & set(D['C']))
print('共同题目:', len(common))

print()
print('=' * 84)
print('A/B/C 在共同前缀上的 V 一致性（前 32 步）')
print('=' * 84)

pairs = [('A','B'), ('A','C'), ('B','C')]
for ka, kb in pairs:
    mads, maxads, pears, exacts = [], [], [], []
    for q in common:
        va = np.array(D[ka][q]['V'][:32], float)
        vb = np.array(D[kb][q]['V'][:32], float)
        L = min(len(va), len(vb))
        if L < 2: continue
        va, vb = va[:L], vb[:L]
        d = np.abs(va - vb)
        mads.append(d.mean()); maxads.append(d.max())
        exacts.append((d == 0).all())
        if va.std() > 1e-12 and vb.std() > 1e-12:
            pears.append(np.corrcoef(va, vb)[0, 1])
    mads = np.array(mads); maxads = np.array(maxads); pears = np.array(pears)
    print(f'--- {ka} vs {kb} (n={len(mads)}) ---')
    print('  MAD   : mean=%.3e  median=%.3e  p90=%.3e' % (mads.mean(), np.median(mads), np.percentile(mads, 90)))
    print('  MaxAD : mean=%.3e  median=%.3e  p90=%.3e  max=%.3e' % (maxads.mean(), np.median(maxads), np.percentile(maxads, 90), maxads.max()))
    print('  Pearson: mean=%.4f  median=%.4f  min=%.4f' % (pears.mean(), np.median(pears), pears.min()))
    print('  完全一致题目比例: %.1f%%  (%d/%d)' % (100*np.mean(exacts), sum(exacts), len(exacts)))
    print('  MAD < 1e-3 的题目比例: %.1f%%' % (100*np.mean(mads < 1e-3)))
    print('  MAD < 1e-2 的题目比例: %.1f%%' % (100*np.mean(mads < 1e-2)))
    print()

# ---- 差异是否与长度相关 ----
print('=' * 84)
print('差异与轨迹长度的关系（A vs C，前 32 步）')
print('=' * 84)
mads_by_len = []
for q in common:
    va = np.array(D['A'][q]['V'][:32], float)
    vc = np.array(D['C'][q]['V'][:32], float)
    L = min(len(va), len(vc))
    if L < 2: continue
    mad = np.abs(va[:L] - vc[:L]).mean()
    mads_by_len.append((D['A'][q]['ep_len'], mad))
arr = np.array(mads_by_len)
print('corr(A轨迹长度, MAD) = %.4f  (n=%d)' % (np.corrcoef(arr[:,0], arr[:,1])[0,1], len(arr)))

# ---- 是否与 step 位置相关 ----
print()
print('=' * 84)
print('V 差异随 step 位置的变化（A vs C，逐位置平均 |ΔV|）')
print('=' * 84)
per_pos = []
for t in range(32):
    diffs = []
    for q in common:
        va = D['A'][q]['V']; vc = D['C'][q]['V']
        if t < len(va) and t < len(vc):
            diffs.append(abs(va[t] - vc[t]))
    if diffs:
        per_pos.append((t, np.mean(diffs), len(diffs)))
print('step  mean|ΔV|   n')
for t, m, n in per_pos[::4]:
    print('%4d  %.3e  %d' % (t, m, n))

# ---- V 在位置上是否单调递增（H2 位置效应检验）----
print()
print('=' * 84)
print('H2 位置效应检验：V 是否随 step 位置系统性上升（A 档，前 32 步）')
print('=' * 84)
slopes = []
for q in common:
    va = np.array(D['A'][q]['V'][:32], float)
    if len(va) >= 8:
        x = np.arange(len(va))
        slopes.append(np.polyfit(x, va, 1)[0])
slopes = np.array(slopes)
print('每步 V 斜率: mean=%+.3e  median=%+.3e' % (slopes.mean(), np.median(slopes)))
print('斜率为正的题目比例: %.1f%%' % (100 * np.mean(slopes > 0)))

# 同样看 final 附近
print()
print('=' * 84)
print('V 在轨迹末段的行为（A 档，最后 10 步 vs 开头 10 步）')
print('=' * 84)
head_means, tail_means, finals = [], [], []
for q in common:
    va = np.array(D['A'][q]['V'], float)
    if len(va) >= 20:
        head_means.append(va[:10].mean())
        tail_means.append(va[-10:].mean())
        finals.append(va[-1])
print('开头10步均值: %.4f   末尾10步均值: %.4f   最后一步: %.4f' % (
    np.mean(head_means), np.mean(tail_means), np.mean(finals)))
print('末尾/开头 比值: %.2f' % (np.mean(tail_means) / max(np.mean(head_means), 1e-9)))
