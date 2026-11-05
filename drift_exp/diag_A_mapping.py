#!/usr/bin/env python3
# P0 诊断：A 档 critic 与 solver 的长度映射异常
import json, hashlib, pathlib, sys
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

print('=' * 78)
print('STEP 1: 文件指纹与修改时间')
print('=' * 78)
files = {
    'A': ROOT / 'drift_exp/solver_A_ms512.jsonl',
    'B': ROOT / 'drift_exp/solver_B_ms64.jsonl',
    'C': ROOT / 'drift_exp/solver_C_ms32.jsonl',
}
import os, datetime
for k, p in files.items():
    st = p.stat()
    print(f'{k}: {p.name}')
    print(f'   sha256={sha256(p)}')
    print(f'   mtime={datetime.datetime.fromtimestamp(st.st_mtime)}')
    print(f'   size={st.st_size}')

print()
print('=' * 78)
print('STEP 2: 各档 ep_len vs solver 步数 的匹配情况')
print('=' * 78)
adv_names = {'A': 'drift_A_ms512', 'B': 'drift_B_ms64', 'C': 'drift_C_ms32'}
d = ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards'

for k in ['A', 'B', 'C']:
    sol = load(files[k])
    adv = load(d / adv_names[k] / f'adv_token_pred-gamma0.95-len32768-{adv_names[k]}.jsonl')
    adv_by = {r['traj_id']: r for r in adv}

    # solver 每条的"步数"候选：
    #   assigned_reward 长度
    #   conversations 消息数
    match_reward = 0
    match_conv = 0
    total = 0
    samples = []
    for tid, r in sorted(adv_by.items()):
        if tid >= len(sol):
            continue
        s = sol[tid]
        ar = len(s.get('assigned_reward') or [])
        cv = len(s.get('conversations') or [])
        ep = r.get('ep_len')
        total += 1
        if ep == ar:
            match_reward += 1
        if ep == cv:
            match_conv += 1
        if len(samples) < 6:
            samples.append((tid, s.get('id'), ep, ar, cv, len(r.get('V') or [])))

    print(f'--- {k} 档 (n={total}) ---')
    print(f'   ep_len == assigned_reward 长度 : {match_reward}/{total}')
    print(f'   ep_len == conversations 消息数  : {match_conv}/{total}')
    print('   tid  qid                  ep_len  reward_len  conv_len  V_len')
    for t in samples:
        print('   %-4s %-22s %-7s %-11s %-9s %s' % t)
    print()

print('=' * 78)
print('STEP 3: A 档 ep_len 与 solver 步数的关系（试图找出变换）')
print('=' * 78)
solA = load(files['A'])
advA = load(d / adv_names['A'] / f'adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
advA_by = {r['traj_id']: r for r in advA}

rows = []
for tid, r in sorted(advA_by.items()):
    if tid >= len(solA):
        continue
    s = solA[tid]
    ar = len(s.get('assigned_reward') or [])
    ep = r.get('ep_len')
    rows.append((tid, ep, ar, len(r.get('V') or [])))

eps = np.array([x[1] for x in rows], dtype=float)
ars = np.array([x[2] for x in rows], dtype=float)
print('A: ep_len 统计  min=%d max=%d mean=%.1f' % (eps.min(), eps.max(), eps.mean()))
print('A: reward_len   min=%d max=%d mean=%.1f' % (ars.min(), ars.max(), ars.mean()))
print('相关系数 corr(ep_len, reward_len) = %.4f' % np.corrcoef(eps, ars)[0, 1])
print('ep_len/reward_len 比值: mean=%.3f median=%.3f' % (np.mean(eps / np.maximum(ars, 1)), np.median(eps / np.maximum(ars, 1))))
print()
print('前 15 条明细 (tid, ep_len, reward_len, V_len, diff):')
for tid, ep, ar, vl in rows[:15]:
    print('   tid=%-4d ep_len=%-5d reward_len=%-5d V_len=%-5d diff=%+d' % (tid, ep, ar, vl, ep - ar))

print()
print('=' * 78)
print('STEP 4: 若按行序(而非 traj_id)对齐，A 档匹配率是否更高？')
print('=' * 78)
# 行序对齐：第 i 条 adv 对第 i 条 solver
advA_ordered = advA
match_seq = 0
for i, r in enumerate(advA_ordered):
    if i >= len(solA):
        break
    ar = len(solA[i].get('assigned_reward') or [])
    if r.get('ep_len') == ar:
        match_seq += 1
print(f'A 档 行序对齐 ep_len==reward_len : {match_seq}/{len(advA_ordered)}')

print()
print('=' * 78)
print('STEP 5: 是否存在旧版 solver_A 快照')
print('=' * 78)
import glob
cands = []
for pat in ['drift_exp/solver_A*', 'drift_exp/*A_ms512*', '**/solver_A_ms512*']:
    cands.extend(glob.glob(str(ROOT / pat), recursive=True))
seen = set()
for c in sorted(set(cands)):
    p = pathlib.Path(c)
    if p.is_file():
        st = p.stat()
        print(f'   {p.relative_to(ROOT)}  size={st.st_size}  mtime={datetime.datetime.fromtimestamp(st.st_mtime)}')
