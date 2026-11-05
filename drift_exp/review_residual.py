#!/usr/bin/env python3
"""
复核 p0_5_residual.py 的实现，重点排查三个潜在问题：
P1: 位置趋势 f(t) 是否有生存偏差（survivorship bias）
P2: final_res 在「各自最后一步」取残差，是否混入长度信息
P3: 控制长度后（固定位置取残差），「Qmin残差>V残差」结论是否稳健

复核方法：改用「固定位置 t 取残差」+「只比较 ep_len>=t 的题」，
彻底消除长度混杂，再看结论是否成立。
"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

det = []
with open(ROOT / 'drift_exp/A_mapping_deterministic.tsv') as f:
    f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        det.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})
Arow = {d['traj_id']: d['solver_row'] for d in det}

solA = load(ROOT / 'drift_exp/solver_A_ms512.jsonl')
advA = load(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl')

data = {}
for r in advA:
    row = Arow.get(r['traj_id'])
    if row is None: continue
    s = solA[row]
    V = r.get('V') or []; Q = r.get('Qmin') or []
    data[s['id']] = {
        'V': np.array(V, float), 'Q': np.array(Q, float),
        'ep_len': r.get('ep_len'),
        'success': bool((s.get('meta') or {}).get('success')),
    }
qids = sorted(data)

def auc_ties(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

# ---------- P1: 生存偏差检验 ----------
print('=' * 78)
print('P1: 位置趋势 f(t) 的生存偏差')
print('=' * 78)
# 比较「全体题目」拟合的 f(t) vs「长轨迹题目(ep_len>=64)」拟合的 f(t)
def fit(all_qids):
    maxlen = max(data[q]['ep_len'] for q in all_qids)
    f = np.zeros(maxlen); cnt = np.zeros(maxlen)
    for q in all_qids:
        seq = data[q]['V']; L = len(seq)
        f[:L] += seq; cnt[:L] += 1
    return np.where(cnt > 0, f / np.maximum(cnt, 1), 0), cnt

f_all, _ = fit(qids)
long_q = [q for q in qids if data[q]['ep_len'] >= 64]
f_long, _ = fit(long_q)
print('  全体题目 n=%d, 长轨迹题目(ep>=64) n=%d' % (len(qids), len(long_q)))
for t in [0, 8, 16, 32, 48, 64]:
    if t < len(f_all) and t < len(f_long):
        print('  t=%-3d  f_all=%.4f  f_long=%.4f  差=%.4f' % (t, f_all[t], f_long[t], f_all[t]-f_long[t]))
print('  → 若 f_all 与 f_long 差异大，说明 f(t) 受「短轨迹在早期就结束」影响，存在生存偏差')

# ---------- P2+P3: 固定位置取残差，控制长度 ----------
print()
print('=' * 78)
print('P2/P3: 固定位置 t 取残差（只比较 ep_len >= t 的题，彻底去长度混杂）')
print('=' * 78)
rng = np.random.default_rng(42)
K = 5
qids_arr = np.array(qids)
rng.shuffle(qids_arr)

for key, label in [('V', 'V'), ('Q', 'Qmin')]:
    print(f'--- {label} ---')
    for t in [16, 32, 48]:
        # 只保留 ep_len >= t 的题
        eligible = [q for q in qids_arr if data[q]['ep_len'] >= t]
        # question-level K-fold 残差
        preds_res = []; preds_raw = []; ys = []
        for fold in range(K):
            test_mask = np.zeros(len(eligible), bool)
            test_mask[fold::K] = True
            test_q = [eligible[i] for i in range(len(eligible)) if test_mask[i]]
            train_q = [eligible[i] for i in range(len(eligible)) if not test_mask[i]]
            # 用训练题拟合全局位置趋势
            f, cnt = fit(train_q)
            for q in test_q:
                seq = data[q][key]
                tt = min(t, len(f) - 1)
                tt = min(tt, len(seq) - 1)
                if tt < 0 or tt >= len(f) or tt >= len(seq):
                    continue
                preds_res.append(seq[tt] - f[tt])   # 固定位置残差
                preds_raw.append(seq[tt])            # 固定位置原始值
                ys.append(1.0 if data[q]['success'] else 0.0)
        ys = np.array(ys)
        a_res = auc_ties([s for s,y in zip(preds_res,ys) if y==1], [s for s,y in zip(preds_res,ys) if y==0])
        a_raw = auc_ties([s for s,y in zip(preds_raw,ys) if y==1], [s for s,y in zip(preds_raw,ys) if y==0])
        n_s = int(ys.sum()); n_f = int(len(ys)-ys.sum())
        print('  t=%-3d n=%-3d(成功%d/失败%d)  原始AUC=%.3f  固定位置残差AUC=%.3f' % (t, len(ys), n_s, n_f, a_raw, a_res))

print()
print('=' * 78)
print('结论判定')
print('=' * 78)
print('  - 若 V 的固定位置残差 AUC 仍≈0.5，而 Qmin 的 >0.6：')
print('    确认「V=位置、Qmin=质量」的分工，且不受长度混杂影响')
