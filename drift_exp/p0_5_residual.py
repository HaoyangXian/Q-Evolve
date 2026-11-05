#!/usr/bin/env python3
"""
P0.5 位置-残差分析（你第六节提出，零 GPU）
核心问题：去除位置趋势后，V/Qmin 的残差是否仍含「状态质量」信息？

严格做法：按题目 ID 分组交叉验证，在训练题目上拟合位置趋势 f(t)，
在测试题目上计算残差，检验残差能否区分「成功题 vs 失败题」。

关键：A 档 temperature=0，每题只有一条轨迹，所以只能做「跨题」的残差分析
（成功题 vs 失败题的残差分布差异），不能做「同题同位置不同质量」。
后者必须等 Branching（P1）。
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# A 档确定性映射
det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})
Arow = {d['traj_id']: d['solver_row'] for d in det}

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')

# 收集 A 档每题的 (V 序列, Qmin 序列, ep_len, 是否成功)
data = {}  # qid -> {V, Q, ep_len, success}
for r in advA:
    row = Arow.get(r['traj_id'])
    if row is None: continue
    s = solA[row]
    qid = s['id']
    V = r.get('V') or []; Q = r.get('Qmin') or []
    data[qid] = {
        'V': np.array(V, float),
        'Q': np.array(Q, float),
        'ep_len': r.get('ep_len'),
        'success': bool((s.get('meta') or {}).get('success')),
    }

qids = sorted(data)
n_succ = sum(1 for q in qids if data[q]['success'])
print('A 档题目:', len(qids), ' 成功:', n_succ, ' 失败:', len(qids) - n_succ)

def auc_ties(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

# ---------- 全局位置趋势 f(t) ----------
# 用「训练题目」拟合：对每个 step t，取所有 ep_len>=t 的题的 V_t 均值
def fit_position_trend(train_qids, key):
    """返回 f(t) 数组，f[t] = 训练题在第 t 步的均值"""
    maxlen = max(data[q]['ep_len'] for q in train_qids)
    f = np.zeros(maxlen)
    cnt = np.zeros(maxlen)
    for q in train_qids:
        seq = data[q][key]
        L = len(seq)
        f[:L] += seq
        cnt[:L] += 1
    return np.where(cnt > 0, f / np.maximum(cnt, 1), 0.0), cnt

def residual_stats(q, key, f):
    seq = data[q][key]
    L = min(len(seq), len(f))
    res = seq[:L] - f[:L]
    return {
        'final_res': res[-1],
        'mean_res': res.mean(),
        'max_res': res.max(),
        'trend_res': res[-1] - res[0] if L > 1 else 0.0,
    }

# ---------- question-level K-fold，检验残差预测成功 ----------
rng = np.random.default_rng(42)
K = 5
qids_arr = np.array(qids)
rng.shuffle(qids_arr)

for key, label in [('V', 'V'), ('Q', 'Qmin')]:
    print()
    print('=' * 78)
    print(f'残差分析（{label}）：去除位置趋势后，能否区分成功/失败题')
    print('=' * 78)
    all_preds = {stat: [] for stat in ['final_res', 'mean_res', 'max_res', 'trend_res']}
    all_y = []
    all_final_raw = []
    all_mean_raw = []

    for fold in range(K):
        test_mask = np.zeros(len(qids_arr), bool)
        test_mask[fold::K] = True
        test_q = qids_arr[test_mask].tolist()
        train_q = qids_arr[~test_mask].tolist()

        f, cnt = fit_position_trend(train_q, key)

        for q in test_q:
            rs = residual_stats(q, key, f)
            for stat in all_preds:
                all_preds[stat].append(rs[stat])
            all_y.append(1.0 if data[q]['success'] else 0.0)
            all_final_raw.append(data[q][key][-1])
            all_mean_raw.append(data[q][key].mean())

    all_y = np.array(all_y)

    def report(name, scores, direction=+1):
        # 成功题 vs 失败题：成功题 score 应该更高
        pos = [direction * s for s, y in zip(scores, all_y) if y == 1]
        neg = [direction * s for s, y in zip(scores, all_y) if y == 0]
        a = auc_ties(pos, neg)
        print('  %-18s AUC(成功vs失败)=%.3f' % (name, a))
        return a

    print('  --- 原始（含位置）---')
    report('final_raw', all_final_raw, +1)
    report('mean_raw', all_mean_raw, +1)
    print('  --- 残差（去位置）---')
    for stat in ['final_res', 'mean_res', 'max_res', 'trend_res']:
        report(stat, all_preds[stat], +1)

# ---------- 位置趋势的形态 ----------
print()
print('=' * 78)
print('位置趋势 f(t) 的形态（用全部 A 档题目拟合，供观察）')
print('=' * 78)
for key, label in [('V', 'V'), ('Q', 'Qmin')]:
    f, cnt = fit_position_trend(qids, key)
    # 采样几个点打印
    print(f'  {label}: f(0)=%+.4f  f(8)=%+.4f  f(16)=%+.4f  f(32)=%+.4f  f(64)=%+.4f  f(128)=%+.4f'
          % (f[0], f[8] if len(f)>8 else np.nan, f[16] if len(f)>16 else np.nan,
             f[32] if len(f)>32 else np.nan, f[64] if len(f)>64 else np.nan, f[128] if len(f)>128 else np.nan))

# ---------- 成功/失败题的 V 轨迹形状对比 ----------
print()
print('=' * 78)
print('成功 vs 失败题的 V 轨迹形状（前 32 步，标准化到同一起点）')
print('=' * 78)
succ_v = [data[q]['V'][:32] for q in qids if data[q]['success'] and len(data[q]['V']) >= 32]
fail_v = [data[q]['V'][:32] for q in qids if not data[q]['success'] and len(data[q]['V']) >= 32]
succ_v = np.array(succ_v); fail_v = np.array(fail_v)
print('  成功题 n=%d, 失败题 n=%d' % (len(succ_v), len(fail_v)))
# 每个位置均值
s_mean = succ_v.mean(0); f_mean = fail_v.mean(0)
print('  step  成功均值   失败均值   差')
for t in [0, 4, 8, 16, 24, 31]:
    if t < len(s_mean):
        print('  %-5d %+.4f   %+.4f   %+.4f' % (t, s_mean[t], f_mean[t], s_mean[t]-f_mean[t]))
