#!/usr/bin/env python3
# 生成题目 ID 对照表 + 用 traj_id 正确对齐的负对照
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
from eval_agent.qa_verifier import answers_equal

d = ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards'
names = {'A': 'drift_A_ms512', 'B': 'drift_B_ms64', 'C': 'drift_C_ms32'}
solvers = {'A': 'solver_A_ms512', 'B': 'solver_B_ms64', 'C': 'solver_C_ms32'}

def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]

# ---------- 1. 构建每档：行索引 -> 题目 id / ep_len / 是否答对 ----------
per = {}
for k, nm in names.items():
    sol = load(ROOT / 'drift_exp' / f'{solvers[k]}.jsonl')
    adv = load(d / nm / f'adv_token_pred-gamma0.95-len32768-{nm}.jsonl')
    adv_by_tid = {r['traj_id']: r for r in adv}
    recs = []
    for idx, s in enumerate(sol):
        a = adv_by_tid.get(idx)
        recs.append({
            'idx': idx,
            'qid': s.get('id'),
            'has_critic': a is not None,
            'ep_len': a.get('ep_len') if a else None,
            'ok': bool(answers_equal(s.get('pred_answer', ''), s.get('gold_answer', ''))),
        })
    per[k] = recs
    print(f'[{k}] solver={len(sol)} critiqued={len(adv)} missing={len(sol)-len(adv)}', file=sys.stderr)

base = load(ROOT / 'data/qa/olympiad_eval200_seed20261031/base.jsonl')
base_ok = {b['id']: bool(answers_equal(b.get('pred_answer',''), b.get('gold_answer',''))) for b in base}

# ---------- 2. 输出题目 ID 对照表 ----------
out = ROOT / 'drift_exp' / 'id_crosswalk.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('qid\tbase_ok\tA_has\tA_len\tA_ok\tB_has\tB_len\tB_ok\tC_has\tC_len\tC_ok\tfull3\tcritic3\n')
    # 以 A 的题目顺序为主
    for r in per['A']:
        qid = r['qid']
        row = [qid, str(base_ok.get(qid, ''))]
        for k in ['A', 'B', 'C']:
            # 按 qid 找（三档题目 id 集合相同）
            m = next((x for x in per[k] if x['qid'] == qid), None)
            if m:
                row += [str(m['has_critic']), str(m['ep_len']), str(int(m['ok']))]
            else:
                row += ['', '', '']
        full3 = all(next((x for x in per[k] if x['qid'] == qid), None) for k in ['A','B','C'])
        critic3 = all((next((x for x in per[k] if x['qid'] == qid), None) or {}).get('has_critic') for k in ['A','B','C'])
        row += [str(int(bool(full3))), str(int(bool(critic3)))]
        f.write('\t'.join(row) + '\n')
print(f'[out] {out}', file=sys.stderr)

# ---------- 3. 正确对齐的负对照（用 traj_id 映射，不靠行序） ----------
def agg(rec):
    V = rec.get('V') or []; Q = rec.get('Qmin') or []; A = rec.get('A') or []
    if not V: return None
    return {'mean_V': sum(V)/len(V), 'final_V': V[-1],
            'mean_Qmin': sum(Q)/len(Q) if Q else float('nan'),
            'mean_A': sum(A)/len(A) if A else float('nan')}

print()
print('=' * 70)
print('负对照修正版：用 traj_id 正确对齐（A 档 = 正常 Solver）')
print('=' * 70)

nm = names['A']
adv = load(d / nm / f'adv_token_pred-gamma0.95-len32768-{nm}.jsonl')
adv_by_tid = {r['traj_id']: r for r in adv}
recsA = per['A']

groups = {}
for k in ['mean_V', 'final_V', 'mean_Qmin', 'mean_A']:
    easy, hard = [], []
    for r in recsA:
        a = adv_by_tid.get(r['idx'])
        if a is None: continue
        ag = agg(a)
        if ag is None: continue
        bok = base_ok.get(r['qid'])
        if bok is None: continue
        (easy if bok else hard).append(ag[k])
    groups[k] = (easy, hard)
    if easy and hard:
        me, mh = np.mean(easy), np.mean(hard)
        print(f'  {k:<12} 易题={me:.4f}(n={len(easy)})  难题={mh:.4f}(n={len(hard)})  差={abs(me-mh):.4f}')

print()
print('  说明：原 calc_drift_auc.py 用 list(zip(crits,sols)) 行序对齐，')
print('        但 adv 输出按长度排序，行序≠题目顺序，故原负对照分组错位。')
print('        本表用 traj_id(=solver 行索引) 正确映射。')
