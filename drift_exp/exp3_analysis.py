#!/usr/bin/env python3
"""Experiment 3 结果分析：判定修正 + 量级对比 + 与实际数据的关联"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')

vals = {}
for line in open(ROOT / 'drift_exp/exp3_termination_results.jsonl'):
    r = json.loads(line)
    vals[(r['qid'], r['pos'], r['cond'])] = r['V']

keys = [(q, p) for (q, p, c) in vals if c == 'E1_normal']
print('有效组合:', len(keys))

print()
print('=' * 80)
print('判定修正：用 median + 完全一致比例（排除浮点非确定性）')
print('=' * 80)
print('%-14s %12s %12s %14s %14s' % ('条件', 'median ΔV', 'mean|ΔV|', '完全一致%', '<1e-4 比例'))
for cond in ['E2_eos', 'E3_trunc', 'E4_continue', 'E5_ast_open']:
    dv = np.array([vals[(q, p, cond)] - vals[(q, p, 'E1_normal')]
                   for (q, p) in keys if (q, p, cond) in vals])
    print('%-14s %+12.6f %12.6f %13.1f%% %13.1f%%' % (
        cond, np.median(dv), np.abs(dv).mean(),
        100*np.mean(np.abs(dv) < 1e-6), 100*np.mean(np.abs(dv) < 1e-4)))

print()
print('=' * 80)
print('量级对比：终止标记效应 vs A/B/C 档间真实差异')
print('=' * 80)
# A/B/C 档的 mean_V（来自之前分析）
tier_meanV = {'A': 0.0355, 'B': 0.0292, 'C': 0.0174}
print('A/B/C 三档 mean_V: A=%.4f B=%.4f C=%.4f' % (tier_meanV['A'], tier_meanV['B'], tier_meanV['C']))
print('  A-C 差 = %.4f' % (tier_meanV['A'] - tier_meanV['C']))
print('  A-B 差 = %.4f' % (tier_meanV['A'] - tier_meanV['B']))

dv_trunc = np.array([vals[(q, p, 'E3_trunc')] - vals[(q, p, 'E1_normal')]
                     for (q, p) in keys if (q, p, 'E3_trunc') in vals])
print()
print('终止标记 (去 <|im_end|>) 造成的 |ΔV| = %.4f' % abs(dv_trunc.mean()))
print('  → 是 A-C 档差的 %.1f 倍' % (abs(dv_trunc.mean()) / (tier_meanV['A'] - tier_meanV['C'])))
print('  → 是 A-B 档差的 %.1f 倍' % (abs(dv_trunc.mean()) / (tier_meanV['A'] - tier_meanV['B'])))

print()
print('=' * 80)
print('实际数据中 trunc 效应的来源：自然结束 vs 跑满预算')
print('=' * 80)

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
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')

natural, truncated = [], []
for r in advA:
    if not r.get('V'): continue
    row = A_row.get(r['traj_id'])
    if row is None: continue
    fv = r['V'][-1]
    ep = r['ep_len']
    (truncated if ep >= 512 else natural).append((fv, ep, solA[row]['id']))

print('A 档内部（max_steps=512）:')
print('  自然结束 (ep<512): n=%d  final_V mean=%+.4f' % (len(natural), np.mean([x[0] for x in natural])))
print('  跑满预算 (ep=512): n=%d  final_V mean=%+.4f' % (len(truncated), np.mean([x[0] for x in truncated])))
if natural and truncated:
    d = np.mean([x[0] for x in natural]) - np.mean([x[0] for x in truncated])
    print('  差 = %+.4f' % d)
    print()
    print('  → A 档内部仅凭「是否跑满预算」，final_V 就有 %.4f 的差距' % abs(d))
    print('  → 这解释了 Experiment 1 中 trunc 单独 AUC=0.720')
    print()
    # 检查自然结束的轨迹末步是否含 FINAL_ANSWER
    n_fa = 0
    for fv, ep, qid in natural:
        s = next((x for x in solA if x['id'] == qid), None)
        if s:
            conv = s.get('conversations') or []
            if conv and 'FINAL_ANSWER' in str(conv[-1].get('value', '')):
                n_fa += 1
    print('  自然结束轨迹中，末步含 <FINAL_ANSWER> 的比例: %d/%d = %.1f%%' % (
        n_fa, len(natural), 100*n_fa/max(len(natural),1)))
