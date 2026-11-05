#!/usr/bin/env python3
# 长度混杂控制：控制 ep_len 后 critic 是否仍有区分能力（偏相关 / 分层）
import json, pathlib, sys
import numpy as np

sys.path.insert(0, '/data3/xhy/codes/work/paper_repos/Q-Evolve')
drift = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/advantage_inference/qa_traj_tokens/wo_assigned_rewards')
names = ['drift_A_ms512', 'drift_B_ms64', 'drift_C_ms32']

def load(name):
    p = drift / name / f'adv_token_pred-gamma0.95-len32768-{name}.jsonl'
    return [json.loads(x) for x in p.open() if x.strip()]

def agg(rec):
    V = rec.get('V') or []; Q = rec.get('Qmin') or []; A = rec.get('A') or []
    if not V: return None
    return {'mean_V': sum(V)/len(V), 'final_V': V[-1],
            'mean_Qmin': sum(Q)/len(Q) if Q else float('nan'),
            'mean_A': sum(A)/len(A) if A else float('nan'),
            'ep_len': rec.get('ep_len')}

def auc(scores, labels):
    pos = sorted(s for s,l in zip(scores,labels) if l==1)
    neg = sorted(s for s,l in zip(scores,labels) if l==0)
    if not pos or not neg: return float('nan')
    nn=len(neg); s=0.0
    for p in pos:
        lo,hi=0,nn
        while lo<hi:
            mid=(lo+hi)//2
            if neg[mid]<p: lo=mid+1
            else: hi=mid
        s+=lo
    return s/(len(pos)*nn)

keys = ['mean_V','final_V','mean_Qmin','mean_A']
rows_by_g = []
for name in names:
    g=[a for a in (agg(r) for r in load(name)) if a and a['ep_len'] is not None]
    rows_by_g.append(g)

print('='*70)
print('5) 长度控制后的条件区分力（只在 A/B/C 重叠的 ep_len 区间比较）')
print('   用 7..32 步重叠窗口（三档都有样本的长度范围）')
print('='*70)
# 三档 ep_len 重叠区：A min=7 max=512, B 7..64, C 7..32 -> 重叠 7..32
lo_len, hi_len = 7, 32
for k in keys:
    scores=[]; labels=[]
    for gi,g in enumerate(rows_by_g):
        for a in g:
            if lo_len <= a['ep_len'] <= hi_len:
                scores.append(a[k]); labels.append(0 if gi==0 else 1)
    if scores:
        a=auc(scores,labels)
        print(f'  窗口[{lo_len},{hi_len}] {k:<12} AUC={a:.3f} 反向={1-a:.3f} n={len(scores)}')

print()
print('='*70)
print('6) 用 ep_len 本身做检测的 AUC（对照：长度是不是更强的信号）')
print('='*70)
for gi, name in enumerate(names):
    scores=[a['ep_len'] for a in rows_by_g[gi]]
    labels=[0]*len(scores)
    # A vs BC
all_lens=[]; all_labels=[]
for gi,g in enumerate(rows_by_g):
    for a in g:
        all_lens.append(a['ep_len']); all_labels.append(0 if gi==0 else 1)
a=auc(all_lens, all_labels)
print(f'  ep_len 直接作为检测信号：AUC={a:.3f} 反向={1-a:.3f}')

print()
print('='*70)
print('7) 控制 ep_len 的偏相关：critic 指标在去除长度线性影响后的残差 AUC')
print('='*70)
# 对 log(ep_len) 做线性回归，取残差，再算 AUC
all_data=[]
for gi,g in enumerate(rows_by_g):
    for a in g:
        all_data.append((gi,a))
lens=np.log(np.array([a['ep_len'] for _,a in all_data]))
for k in keys:
    vals=np.array([a[k] for _,a in all_data])
    # 简单线性回归
    A=np.vstack([lens, np.ones(len(lens))]).T
    coef,_,_,_=np.linalg.lstsq(A, vals, rcond=None)
    resid=vals - A@coef
    labels=np.array([gi for gi,_ in all_data])
    a=auc(resid.tolist(), labels.tolist())
    # 注意残差方向：退化更低则 AUC<0.5
    print(f'  残差(去除log len) {k:<12} AUC={a:.3f} 反向={1-a:.3f}')
