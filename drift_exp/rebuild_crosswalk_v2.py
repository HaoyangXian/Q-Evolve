#!/usr/bin/env python3
# 重建 id_crosswalk（用 LCS 恢复的 A 档映射 + meta.success 标签）
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---- LCS 恢复 A 映射 ----
solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
rlA = [len(s.get('assigned_reward') or []) for s in solA]
adv_sorted = sorted(advA, key=lambda r: r['traj_id'])
eps = [r['ep_len'] for r in adv_sorted]
n, m = len(eps), len(rlA)
dp = np.zeros((n+1, m+1), dtype=np.int32)
for i in range(n-1, -1, -1):
    for j in range(m-1, -1, -1):
        dp[i][j] = dp[i+1][j+1] + 1 if eps[i] == rlA[j] else max(dp[i+1][j], dp[i][j+1])
i = j = 0
Amap = {}
while i < n and j < m:
    if eps[i] == rlA[j]:
        Amap[adv_sorted[i]['traj_id']] = j
        i += 1; j += 1
    elif dp[i+1][j] >= dp[i][j+1]:
        i += 1
    else:
        j += 1

# ---- 各档：qid -> 记录 ----
def build(k, solver_file, adv_name):
    sol = load(ROOT / f'drift_exp/{solver_file}.jsonl')
    adv = load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{adv_name}/adv_token_pred-gamma0.95-len32768-{adv_name}.jsonl')
    out = {}
    for r in adv:
        tid = r['traj_id']
        row = Amap.get(tid) if k == 'A' else tid
        if row is None or row >= len(sol):
            continue
        s = sol[row]
        ms = (s.get('meta') or {}).get('success')
        out[s['id']] = {
            'row': row,
            'traj_id': tid,
            'ep_len': r.get('ep_len'),
            'reward_len': len(s.get('assigned_reward') or []),
            'ok_eq': bool(answers_equal(s.get('pred_answer',''), s.get('gold_answer',''))),
            'ok_meta': bool(ms),
        }
    return out

data = {
    'A': build('A', 'solver_A_ms512', 'drift_A_ms512'),
    'B': build('B', 'solver_B_ms64', 'drift_B_ms64'),
    'C': build('C', 'solver_C_ms32', 'drift_C_ms32'),
}
base = load(ROOT / 'data/qa/olympiad_eval200_seed20261031/base.jsonl')
base_ok = {b['id']: bool(answers_equal(b.get('pred_answer',''), b.get('gold_answer',''))) for b in base}
base_ms = {b['id']: bool((b.get('meta') or {}).get('success')) for b in base}

allq = sorted(data['A'])
out = ROOT / 'drift_exp' / 'id_crosswalk_v2.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('qid\tbase_ok\tbase_ms\tA_row\tA_tid\tA_len\tA_ok\tA_ms\tB_row\tB_tid\tB_len\tB_ok\tB_ms\tC_row\tC_tid\tC_len\tC_ok\tC_ms\thas_all3\n')
    for q in allq:
        row = [q, str(base_ok.get(q)), str(base_ms.get(q))]
        for k in ['A','B','C']:
            d = data[k].get(q)
            if d:
                row += [str(d['row']), str(d['traj_id']), str(d['ep_len']), str(int(d['ok_eq'])), str(int(d['ok_meta']))]
            else:
                row += ['','','','','']
        row.append(str(int(all(q in data[k] for k in 'ABC'))))
        f.write('\t'.join(row) + '\n')

print('写入:', out)
print('覆盖题目数:', len(allq))
print()
print('校验：A 档 ep_len 与 reward_len 一致率')
ok = sum(1 for q in data['A'] if data['A'][q]['ep_len'] == data['A'][q]['reward_len'])
print('  A: %d/%d' % (ok, len(data['A'])))
for k in ['B','C']:
    ok = sum(1 for q in data[k] if data[k][q]['ep_len'] == data[k][q]['reward_len'])
    print('  %s: %d/%d' % (k, ok, len(data[k])))
print()
print('base_ok vs A_ok 一致:',
      sum(1 for q in data['A'] if base_ok.get(q) == data['A'][q]['ok_eq']), '/', len(data['A']))
print('base_ms vs A_ms 一致:',
      sum(1 for q in data['A'] if base_ms.get(q) == data['A'][q]['ok_meta']), '/', len(data['A']))
