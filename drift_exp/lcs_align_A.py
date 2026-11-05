#!/usr/bin/env python3
# 用 LCS 序列对齐精确恢复 A 档 traj_id -> solver 行 的映射
import json, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')

# 按 traj_id 排序的 ep_len 序列
adv_sorted = sorted(advA, key=lambda r: r['traj_id'])
adv_eps = [r['ep_len'] for r in adv_sorted]
adv_tids = [r['traj_id'] for r in adv_sorted]

reward_len = [len(s.get('assigned_reward') or []) for s in solA]
qids = [s.get('id') for s in solA]

print('adv 条数=%d  traj_id 范围=[%d,%d]' % (len(adv_sorted), min(adv_tids), max(adv_tids)))
print('solver 条数=%d' % len(solA))
print()

# ---------- LCS 对齐 ----------
# 目标：找 adv_eps 在 reward_len 中的最大匹配子序列，保持顺序
n, m = len(adv_eps), len(reward_len)
# DP（n*m = 189*200 很小）
dp = np.zeros((n + 1, m + 1), dtype=np.int32)
for i in range(n - 1, -1, -1):
    for j in range(m - 1, -1, -1):
        if adv_eps[i] == reward_len[j]:
            dp[i][j] = dp[i + 1][j + 1] + 1
        else:
            dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

# 回溯
matches = []  # (adv_index_in_sorted, solver_row)
i = j = 0
while i < n and j < m:
    if adv_eps[i] == reward_len[j]:
        matches.append((i, j))
        i += 1
        j += 1
    elif dp[i + 1][j] >= dp[i][j + 1]:
        i += 1
    else:
        j += 1

print('=' * 78)
print('LCS 对齐结果')
print('=' * 78)
print('匹配对数: %d / %d' % (len(matches), n))
print('LCS 长度: %d' % dp[0][0])

matched_solver_rows = set(r for _, r in matches)
skipped = [r for r in range(len(solA)) if r not in matched_solver_rows]
print()
print('未被匹配的 solver 行（即 critic 没打分的题）: %d 条' % len(skipped))
for r in skipped:
    print('  行 %-3d qid=%-18s reward_len=%d' % (r, qids[r], reward_len[r]))

print()
print('=' * 78)
print('映射样例（前 40 条 adv）')
print('=' * 78)
print('adv_tid  adv_idx  -> solver_row  qid                 ep_len  reward_len  offset')
mismatch = 0
for k, (ai, sr) in enumerate(matches[:40]):
    tid = adv_tids[ai]
    print('%-7d %-7d -> %-11d %-20s %-7d %-11d %+d' % (
        tid, ai, sr, qids[sr], adv_eps[ai], reward_len[sr], sr - tid))
for ai, sr in matches:
    if adv_eps[ai] != reward_len[sr]:
        mismatch += 1
print()
print('校验：LCS 匹配中长度不一致的数量 =', mismatch)

# 完整映射表
print()
print('=' * 78)
print('完整映射（traj_id -> solver 行 -> qid）')
print('=' * 78)
mapping = {}
for ai, sr in matches:
    mapping[adv_tids[ai]] = sr

# 检查映射是否单调（应该严格递增）
rows_seq = [mapping[t] for t in sorted(mapping)]
mono = all(rows_seq[i] < rows_seq[i + 1] for i in range(len(rows_seq) - 1))
print('映射是否严格单调递增:', mono)
print('映射覆盖 traj_id 数:', len(mapping))

# 输出到文件
out = ROOT / 'drift_exp' / 'A_traj_to_solver_mapping.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('traj_id\tsolver_row\tqid\tep_len\treward_len\n')
    for t in sorted(mapping):
        sr = mapping[t]
        f.write('%d\t%d\t%s\t%d\t%d\n' % (t, sr, qids[sr], adv_eps[adv_tids.index(t)], reward_len[sr]))
print('已写入:', out)
