#!/usr/bin/env python3
# 验证 A 档分段偏移修复假设 + ties-aware AUC + 前缀关系 + base vs A
import json, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
advA_by = {r['traj_id']: r for r in advA}
reward_len = [len(s.get('assigned_reward') or []) for s in solA]

print('=' * 78)
print('STEP 1: 验证分段偏移假设  adv[tid] -> solver[tid if tid<31 else tid+1]')
print('=' * 78)

def map_old_to_new(tid):
    return tid if tid < 31 else tid + 1

hits = 0
total = 0
misses = []
for tid in sorted(advA_by.keys()):
    j = map_old_to_new(tid)
    if j >= len(solA):
        misses.append((tid, 'out_of_range'))
        continue
    total += 1
    if advA_by[tid]['ep_len'] == reward_len[j]:
        hits += 1
    else:
        misses.append((tid, advA_by[tid]['ep_len'], reward_len[j]))

print('匹配: %d/%d' % (hits, total))
if misses[:10]:
    print('不匹配样例(前10):', misses[:10])
else:
    print('★ 全部匹配！分段偏移假设成立')

print()
print('=' * 78)
print('STEP 2: 因此 critic A 实际覆盖了 solver_A 的哪些行？哪些缺失？')
print('=' * 78)
covered = set(map_old_to_new(t) for t in advA_by.keys() if map_old_to_new(t) < len(solA))
missing = [i for i in range(len(solA)) if i not in covered]
print('覆盖行数:', len(covered), ' solver 总行数:', len(solA))
print('未覆盖的 solver 行号:', missing)
print()
for i in missing:
    s = solA[i]
    print('  行 %-3d qid=%-18s reward_len=%d' % (i, s.get('id'), reward_len[i]))

print()
print('=' * 78)
print('STEP 3: solver[31] 与 2666 的关系（fix2666 操作）')
print('=' * 78)
print('solver[31].id =', solA[31].get('id'), ' reward_len =', reward_len[31])
print('solver[199].id =', solA[199].get('id'), ' reward_len =', reward_len[199])

# 检查 solver[31] 是否与 adv 里某条内容重复（说明是后来补的）
print()
print('=' * 78)
print('STEP 4: ties-aware AUC（标准定义，含 0.5 权重）')
print('=' * 78)

def auc_ties(scores, labels):
    """标准 AUC: P(S1>S0) + 0.5*P(S1=S0)"""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float('nan')
    pos = np.array(pos); neg = np.array(neg)
    # 向量化
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

def agg(rec):
    V = rec.get('V') or []; Q = rec.get('Qmin') or []; A = rec.get('A') or []
    if not V: return None
    return {'mean_V': sum(V)/len(V), 'final_V': V[-1],
            'mean_Qmin': sum(Q)/len(Q) if Q else float('nan'),
            'mean_A': sum(A)/len(A) if A else float('nan'),
            'ep_len': rec.get('ep_len')}

names = ['drift_A_ms512', 'drift_B_ms64', 'drift_C_ms32']
rows_g = []
for nm in names:
    g = [a for a in (agg(r) for r in load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{nm}/adv_token_pred-gamma0.95-len32768-{nm}.jsonl')) if a]
    rows_g.append(g)
    print(f'[load] {nm}: n={len(g)}')

print()
print('分布级 AUC（正类=Degraded，ties-aware）：')
print('%-12s %10s %14s' % ('指标', '原始AUC', '方向修正'))
for k in ['mean_V', 'final_V', 'mean_Qmin', 'mean_A']:
    scores = []; labels = []
    for gi, g in enumerate(rows_g):
        for a in g:
            scores.append(a[k]); labels.append(0 if gi == 0 else 1)
    a = auc_ties(scores, labels)
    # V/Qmin/A 方向：V/Qmin 越低越退化 → 取负；A 越高越退化 → 保持
    if k in ('mean_V', 'final_V', 'mean_Qmin'):
        corrected = 1 - a
        note = '(取负)'
    else:
        corrected = a
        note = '(保持)'
    print('%-12s %10.3f %14.3f %s' % (k, a, corrected, note))

print()
print('=' * 78)
print('STEP 5: raw accuracy / meta.success 的 ties-aware AUC')
print('=' * 78)
from eval_agent.qa_verifier import answers_equal

sols = {}
for k, nm in zip(['A','B','C'], ['solver_A_ms512','solver_B_ms64','solver_C_ms32']):
    sols[k] = load(ROOT / f'drift_exp/{nm}.jsonl')

# 定义1: answers_equal 错误率
sc1, lb1 = [], []
# 定义2: meta.success 失败率
sc2, lb2 = [], []
for gi, k in enumerate(['A','B','C']):
    for r in sols[k]:
        ok = bool(answers_equal(r.get('pred_answer',''), r.get('gold_answer','')))
        sc1.append(0.0 if ok else 1.0)  # 错误率，越高越退化
        lb1.append(0 if gi==0 else 1)
        ms = (r.get('meta') or {}).get('success')
        sc2.append(0.0 if ms else 1.0)
        lb2.append(0 if gi==0 else 1)

a1 = auc_ties(sc1, lb1)
a2 = auc_ties(sc2, lb2)
print('错误率(answers_equal)   AUC = %.3f  (n=%d)' % (a1, len(sc1)))
print('失败率(meta.success)    AUC = %.3f  (n=%d)' % (a2, len(sc2)))

print()
print('各档正确率对比:')
for k in ['A','B','C']:
    rows = sols[k]
    acc1 = sum(1 for r in rows if answers_equal(r.get('pred_answer',''), r.get('gold_answer',''))) / len(rows)
    acc2 = sum(1 for r in rows if (r.get('meta') or {}).get('success')) / len(rows)
    # false positive: answers_equal True 但 not success
    fp = sum(1 for r in rows if answers_equal(r.get('pred_answer',''), r.get('gold_answer','')) and not (r.get('meta') or {}).get('success'))
    print('  %s: answers_equal=%.3f  meta.success=%.3f  false_positive=%d' % (k, acc1, acc2, fp))

print()
print('=' * 78)
print('STEP 6: base vs solver_A 的一致性')
print('=' * 78)
base = load(ROOT / 'data/qa/olympiad_eval200_seed20261031/base.jsonl')
base_by = {b['id']: b for b in base}
same_pred = 0; same_ok = 0; n = 0
for s in solA:
    b = base_by.get(s['id'])
    if not b: continue
    n += 1
    if s.get('pred_answer') == b.get('pred_answer'): same_pred += 1
    ok_s = bool(answers_equal(s.get('pred_answer',''), s.get('gold_answer','')))
    ok_b = bool(answers_equal(b.get('pred_answer',''), b.get('gold_answer','')))
    if ok_s == ok_b: same_ok += 1
print('base 与 solver_A 共同题目数:', n)
print('pred_answer 完全相同: %d/%d' % (same_pred, n))
print('正确性标签一致    : %d/%d' % (same_ok, n))

print()
print('=' * 78)
print('STEP 7: A/B/C 是否同一条轨迹的前缀关系')
print('=' * 78)
solB = sols['B']; solC = sols['C']
def convs(rows):
    return {r['id']: [m.get('content','') for m in (r.get('conversations') or [])] for r in rows}

cA, cB, cC = convs(solA), convs(solB), convs(solC)
common = set(cA) & set(cB) & set(cC)
print('三档共同题目数:', len(common))

def is_prefix(shorter, longer):
    return len(shorter) <= len(longer) and longer[:len(shorter)] == shorter

pb = pa = pc = 0
for q in common:
    if is_prefix(cB[q], cA[q]): pb += 1
    if is_prefix(cC[q], cA[q]): pa += 1
    if is_prefix(cC[q], cB[q]): pc += 1
print('B 是 A 的前缀: %d/%d' % (pb, len(common)))
print('C 是 A 的前缀: %d/%d' % (pa, len(common)))
print('C 是 B 的前缀: %d/%d' % (pc, len(common)))
