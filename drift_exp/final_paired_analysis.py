#!/usr/bin/env python3
# 最终分析：用 LCS 恢复的映射做题目级配对 AUC + bootstrap CI + 修正负对照
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

def auc_ties(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    pos = np.asarray(pos, dtype=float); neg = np.asarray(neg, dtype=float)
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

# ---------- 1. 构建 per-question 数据（用 LCS 映射） ----------
# A 档映射
solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')
reward_len_A = [len(s.get('assigned_reward') or []) for s in solA]

adv_sorted = sorted(advA, key=lambda r: r['traj_id'])
adv_eps = [r['ep_len'] for r in adv_sorted]
n, m = len(adv_eps), len(reward_len_A)
dp = np.zeros((n+1, m+1), dtype=np.int32)
for i in range(n-1, -1, -1):
    for j in range(m-1, -1, -1):
        dp[i][j] = dp[i+1][j+1] + 1 if adv_eps[i] == reward_len_A[j] else max(dp[i+1][j], dp[i][j+1])
i = j = 0
A_map = {}  # traj_id -> solver_row
while i < n and j < m:
    if adv_eps[i] == reward_len_A[j]:
        A_map[adv_sorted[i]['traj_id']] = j
        i += 1; j += 1
    elif dp[i+1][j] >= dp[i][j+1]:
        i += 1
    else:
        j += 1

# B/C 映射：traj_id == 行号（已验证 200/200）
def build(k, solver_file, adv_name):
    sol = load(ROOT / f'drift_exp/{solver_file}.jsonl')
    adv = load(ROOT / f'advantage_inference/qa_traj_tokens/wo_assigned_rewards/{adv_name}/adv_token_pred-gamma0.95-len32768-{adv_name}.jsonl')
    out = {}  # qid -> agg
    for r in adv:
        tid = r['traj_id']
        row = A_map[tid] if k == 'A' else tid
        if row >= len(sol): continue
        qid = sol[row].get('id')
        a = agg(r)
        if a is None or qid is None: continue
        out[qid] = a
    return out

data = {
    'A': build('A', 'solver_A_ms512', 'drift_A_ms512'),
    'B': build('B', 'solver_B_ms64', 'drift_B_ms64'),
    'C': build('C', 'solver_C_ms32', 'drift_C_ms32'),
}
print('各档有效题目数: A=%d B=%d C=%d' % (len(data['A']), len(data['B']), len(data['C'])))
common = set(data['A']) & set(data['B']) & set(data['C'])
common = sorted(common)
print('三档共同题目数:', len(common))

keys = ['mean_V', 'final_V', 'mean_Qmin', 'mean_A']
# 方向：V/Qmin 取负，A 保持
DIRECTION = {'mean_V': -1, 'final_V': -1, 'mean_Qmin': -1, 'mean_A': +1}

def pair_auc(g_lo, g_hi, k, qids):
    """g_hi 为正类(更退化)；用 DIRECTION 统一为'分数越高越退化'"""
    pos = [DIRECTION[k] * data[g_hi][q][k] for q in qids]
    neg = [DIRECTION[k] * data[g_lo][q][k] for q in qids]
    return auc_ties(pos, neg)

def boot_pair(g_lo, g_hi, k, qids, B=2000, seed=0):
    """question-level bootstrap"""
    rng = np.random.default_rng(seed)
    qs = np.array(qids)
    ests = []
    for _ in range(B):
        idx = rng.integers(0, len(qs), len(qs))
        sub = qs[idx].tolist()
        v = pair_auc(g_lo, g_hi, k, sub)
        if not np.isnan(v):
            ests.append(v)
    if not ests:
        return (np.nan, np.nan)
    return tuple(np.percentile(ests, [2.5, 97.5]))

print()
print('=' * 82)
print('题目级配对 AUC（正类=更退化；已按指标方向统一为"越高越退化"）')
print('n = %d 道共同题' % len(common))
print('=' * 82)
print('%-12s %-14s %10s %22s' % ('指标', '对比', 'AUC', '95%CI(question-bootstrap)'))
for k in keys:
    for lo, hi, label in [('A','B','A-vs-B'), ('A','C','A-vs-C'), ('B','C','B-vs-C')]:
        a = pair_auc(lo, hi, k, common)
        lo_ci, hi_ci = boot_pair(lo, hi, k, common)
        print('%-12s %-14s %10.3f   [%.3f, %.3f]' % (k, label, a, lo_ci, hi_ci))
    print()

print('=' * 82)
print('总体：A(正常) vs B+C(合并退化)')
print('=' * 82)
print('%-12s %10s %22s' % ('指标', 'AUC', '95%CI'))
for k in keys:
    pos = [DIRECTION[k] * data[g][q][k] for g in ('B','C') for q in common]
    neg = [DIRECTION[k] * data['A'][q][k] for q in common]
    a = auc_ties(pos, neg)
    # bootstrap
    rng = np.random.default_rng(1)
    qs = np.array(common); ests = []
    for _ in range(2000):
        idx = rng.integers(0, len(qs), len(qs)); sub = qs[idx].tolist()
        p = [DIRECTION[k] * data[g][q][k] for g in ('B','C') for q in sub]
        nn = [DIRECTION[k] * data['A'][q][k] for q in sub]
        v = auc_ties(p, nn)
        if not np.isnan(v): ests.append(v)
    lo_ci, hi_ci = np.percentile(ests, [2.5, 97.5])
    print('%-12s %10.3f   [%.3f, %.3f]' % (k, a, lo_ci, hi_ci))

print()
print('=' * 82)
print('修正负对照：独立难度（用 base 之外的信号不可得，此处报告 base 与 A 的重合度）')
print('=' * 82)
base = load(ROOT / 'data/qa/olympiad_eval200_seed20261031/base.jsonl')
base_ok = {b['id']: bool(answers_equal(b.get('pred_answer',''), b.get('gold_answer',''))) for b in base}
A_ok = {s['id']: bool(answers_equal(s.get('pred_answer',''), s.get('gold_answer',''))) for s in solA}
agree = sum(1 for q in base_ok if q in A_ok and base_ok[q] == A_ok[q])
tot = sum(1 for q in base_ok if q in A_ok)
print('base_ok 与 A_ok 一致: %d/%d  (%.1f%%)' % (agree, tot, 100*agree/tot))
print('→ base 基本就是 A，负对照不独立，必须换难度来源')

print()
print('按 base 分组的 A 档指标（这是 A 内部对错差异，非难度效应）:')
easy = [q for q in data['A'] if base_ok.get(q) is True]
hard = [q for q in data['A'] if base_ok.get(q) is False]
for k in keys:
    me = np.mean([data['A'][q][k] for q in easy]) if easy else np.nan
    mh = np.mean([data['A'][q][k] for q in hard]) if hard else np.nan
    print('  %-12s 易=%+.4f(n=%d)  难=%+.4f(n=%d)  差=%.4f' % (k, me, len(easy), mh, len(hard), abs(me-mh)))
