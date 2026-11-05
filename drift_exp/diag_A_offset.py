#!/usr/bin/env python3
# 定位 A 档错位边界与偏移量
import json, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
advA_by = {r['traj_id']: r for r in advA}

reward_len = [len(s.get('assigned_reward') or []) for s in solA]
qids = [s.get('id') for s in solA]

print('=' * 78)
print('STEP A: 逐条匹配状态（前 60 条）')
print('=' * 78)
first_bad = None
match_seq = []
for tid in range(189):
    r = advA_by.get(tid)
    if r is None:
        continue
    ep = r.get('ep_len')
    rl = reward_len[tid]
    ok = (ep == rl)
    match_seq.append((tid, ok, ep, rl))
    if not ok and first_bad is None:
        first_bad = tid

print('首个不匹配 tid =', first_bad)
print()
print('tid  ok   ep_len  reward_len  diff')
for tid, ok, ep, rl in match_seq[:40]:
    print('%3d  %-4s %-7d %-11d %+d' % (tid, 'OK' if ok else 'BAD', ep, rl, ep - rl))

print()
print('=' * 78)
print('STEP B: 匹配状态的分段统计（找断点）')
print('=' * 78)
# 把 189 条按每 20 条一组统计
for start in range(0, 189, 20):
    seg = [m for m in match_seq if start <= m[0] < start + 20]
    if not seg:
        continue
    nok = sum(1 for _, ok, _, _ in seg if ok)
    print('  tid %3d-%3d : 匹配 %2d/%2d' % (start, min(start + 19, 188), nok, len(seg)))

print()
print('=' * 78)
print('STEP C: 测试整体偏移假设  ep_len[tid] == reward_len[tid + k]')
print('=' * 78)
ep_arr = np.array([advA_by[t]['ep_len'] for t in range(189) if t in advA_by])
for k in range(-5, 6):
    hits = 0
    n = 0
    for tid in range(189):
        r = advA_by.get(tid)
        if r is None:
            continue
        j = tid + k
        if 0 <= j < 200:
            n += 1
            if r['ep_len'] == reward_len[j]:
                hits += 1
    print('  offset k=%+d : 匹配 %d/%d' % (k, hits, n))

print()
print('=' * 78)
print('STEP D: 题目 2666 的位置（fix2666 操作对象）')
print('=' * 78)
for i, q in enumerate(qids):
    if q and '2666' in q:
        print('  solver_A 中 qid=%s 位于行 %d, reward_len=%d' % (q, i, reward_len[i]))

print()
print('=' * 78)
print('STEP E: 与 B 档交叉验证（B 档 200/200 通过，可作为可信锚点）')
print('=' * 78)
# 用 qid 做锚：A 的 adv[tid] 如果 claim 是 qid X，那 X 在 solver_A 的实际行号是多少？
solB = load(ROOT / 'drift_exp/solver_B_ms64.jsonl')
print('A 档前 20 条：adv 的 tid 对应 solver_A[tid] 的 qid，与 ep_len 对照')
print('tid  qid_A                ep_len  reward_len_A')
for tid in range(20):
    r = advA_by.get(tid)
    if r is None:
        continue
    print('%3d  %-20s %-7d %d' % (tid, qids[tid], r['ep_len'], reward_len[tid]))

print()
print('=' * 78)
print('STEP F: A 档 adv 中 ep_len 的分布 vs solver_A 的 reward_len 分布')
print('=' * 78)
ep_all = np.array([r['ep_len'] for r in advA])
rl_all = np.array(reward_len)
print('adv  ep_len      : min=%d max=%d mean=%.1f median=%.1f  n512=%d' % (ep_all.min(), ep_all.max(), ep_all.mean(), np.median(ep_all), (ep_all == 512).sum()))
print('solver reward_len: min=%d max=%d mean=%.1f median=%.1f  n512=%d' % (rl_all.min(), rl_all.max(), rl_all.mean(), np.median(rl_all), (rl_all == 512).sum()))
