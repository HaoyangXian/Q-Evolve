#!/usr/bin/env python3
"""
Branching Continuation Pilot —— 分析阶段
1) 用冻结 critic 给每个前缀打分（state_pos = 最后 assistant 消息末尾，与 Exp3 一致）
2) 汇总每个前缀的 continuation 成功率
3) 检验：同题同位置下，V 能否预测后续成功率（H1 vs H2）
4) 控制 token length 的稳健性检验
"""
import json, sys, os, pathlib, glob
import numpy as np
import torch

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'QEvolve'))

from transformers import AutoTokenizer
from eval_agent.data_utils import get_chat_template
from inf_critic_chunked_fast import load_probe_full_state_dict

MODEL_BASE = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/sft_success')
MODEL_CKPT = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B')
OUTDIR = ROOT / 'drift_exp/branch_pilot'
MAX_LENGTH = 32768
ROLES = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]

# ---------- 数据汇总 ----------
trajs = read_jsonl(OUTDIR / 'traj_shard0.jsonl') + read_jsonl(OUTDIR / 'traj_shard1.jsonl')
conts = read_jsonl(OUTDIR / 'cont_shard0.jsonl') + read_jsonl(OUTDIR / 'cont_shard1.jsonl')
print('traj=%d  cont=%d' % (len(trajs), len(conts)))

if not trajs or not conts:
    raise SystemExit('数据不足，等待 Pilot 完成')

# 成功率表：(qid, traj_idx, prefix_steps) -> (succ_rate, n, mean_ntok, mean_gen)
rates = {}
for c in conts:
    k = (c['qid'], c['traj_idx'], c['prefix_steps'])
    rates.setdefault(k, []).append(c)
stat = {}
for k, lst in rates.items():
    stat[k] = {
        'p': float(np.mean([1.0 if x['success'] else 0.0 for x in lst])),
        'n': len(lst),
        'ntok': float(np.mean([x['prefix_ntok'] for x in lst])),
        'finish_rate': float(np.mean([1.0 if x['finished'] else 0.0 for x in lst])),
        'mean_gen': float(np.mean([x['n_generated'] for x in lst])),
    }
print('聚合状态点: %d' % len(stat))

# ---------- critic 打分 ----------
tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True)
tok.pad_token = tok.eos_token
tmpl = get_chat_template(MODEL_BASE)
P_SUF = tmpl.role_prefix_and_suffix

def build_prefix_ids(conv, prefix_steps):
    need = 2 * prefix_steps
    if len(conv) < need:
        return None, None
    ids, acc, last_end = [], 0, None
    for i, s in enumerate(conv[:need]):
        try:
            role = ROLES[s['from']]
        except KeyError:
            return None, None
        if role != ('user' if i % 2 == 0 else 'assistant'):
            return None, None
        pre, suf = P_SUF.get(role, ('', ''))
        t = tok(pre + s['value'] + suf, add_special_tokens=False).input_ids
        ids.extend(t); acc += len(t)
        if role == 'assistant':
            last_end = acc - 1
    return ids, last_end

print('[Load] critic...', flush=True)
model, _, resolved = load_probe_full_state_dict(
    base_model_name_or_path=MODEL_BASE, ckpt_dir=MODEL_CKPT,
    device='cuda:0', max_length=MAX_LENGTH, strict=True, use_done_head=False)
model.eval()
print('[Load] done', flush=True)

# 收集所有唯一前缀
items = []
by_key = {(t['qid'], t['traj_idx']): t for t in trajs}
for (qid, ti, ps) in sorted(stat.keys()):
    t = by_key.get((qid, ti))
    if t is None: continue
    ids, spos = build_prefix_ids(t['conversations'], ps)
    if ids is None or spos is None or len(ids) > MAX_LENGTH:
        continue
    items.append({'qid': qid, 'traj_idx': ti, 'ps': ps, 'ids': ids, 'spos': spos})

print('待打分前缀: %d' % len(items))

def run_batch(recs):
    B = len(recs)
    maxL = max(len(r['ids']) for r in recs)
    pad = tok.pad_token_id or tok.eos_token_id
    input_ids = torch.full((B, maxL), pad, dtype=torch.long, device='cuda:0')
    attn = torch.zeros((B, maxL), dtype=torch.long, device='cuda:0')
    state_pos = torch.zeros((B, 1), dtype=torch.long, device='cuda:0')
    for i, r in enumerate(recs):
        L = len(r['ids'])
        input_ids[i, :L] = torch.tensor(r['ids'], dtype=torch.long, device='cuda:0')
        attn[i, :L] = 1
        state_pos[i, 0] = r['spos']
    with torch.no_grad():
        out = model.forward_v_multi(input_ids=input_ids, attention_mask=attn,
                                    state_pos=state_pos, next_pos=state_pos)
        V = out['V'].float().squeeze(-1)[:, 0]
    return V.detach().cpu().numpy()

items.sort(key=lambda r: len(r['ids']))
for i in range(0, len(items), 8):
    b = items[i:i+8]
    v = run_batch(b)
    for r, vv in zip(b, v):
        stat[(r['qid'], r['traj_idx'], r['ps'])]['V'] = float(vv)
    print('  %d/%d' % (i+len(b), len(items)), flush=True)

# ---------- 分析 ----------
rows = []
for k, s in stat.items():
    if 'V' not in s: continue
    rows.append({'qid': k[0], 'traj_idx': k[1], 'ps': k[2], **s})
print('\n可用状态点: %d' % len(rows))

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

print()
print('=' * 76)
print('A) 全局：V vs 后续成功率')
print('=' * 76)
V = [r['V'] for r in rows]; P = [r['p'] for r in rows]
print('  n=%d  V mean=%+.4f  p mean=%.3f' % (len(rows), np.mean(V), np.mean(P)))
print('  Pearson  r=%.3f' % pearson(V, P))
print('  Spearman r=%.3f' % spearman(V, P))
print('  V vs prefix_ntok   r=%.3f' % pearson(V, [r['ntok'] for r in rows]))
print('  p vs prefix_ntok   r=%.3f' % pearson(P, [r['ntok'] for r in rows]))
print('  V vs prefix_steps  r=%.3f' % pearson(V, [r['ps'] for r in rows]))

print()
print('=' * 76)
print('B) 核心检验：同题同位置内（跨 3 条轨迹）V 与成功率的关系')
print('=' * 76)
groups = {}
for r in rows:
    groups.setdefault((r['qid'], r['ps']), []).append(r)

pairs = []
within_corr = []
for (qid, ps), g in groups.items():
    if len(g) < 2: continue
    # 组内两两比较：V 高的轨迹，成功率是否也高
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            dv = g[i]['V'] - g[j]['V']
            dp = g[i]['p'] - g[j]['p']
            if abs(dv) < 1e-9 or abs(dp) < 1e-9:
                continue
            pairs.append((dv, dp))
    if len(g) >= 3 and len(set(x['p'] for x in g)) > 1 and len(set(x['V'] for x in g)) > 1:
        within_corr.append(spearman([x['V'] for x in g], [x['p'] for x in g]))

print('  有效组数(qid,ps): %d' % sum(1 for g in groups.values() if len(g) >= 2))
print('  组内可比较对数: %d' % len(pairs))
if pairs:
    dv = np.array([a for a, b in pairs]); dp = np.array([b for a, b in pairs])
    concord = np.mean((dv > 0) == (dp > 0))
    print('  V 与成功率的符号一致率(concordance): %.3f' % concord)
    print('    （0.5=随机，>0.5 支持 H1）')
within_corr = [c for c in within_corr if not np.isnan(c)]
if within_corr:
    print('  组内 Spearman 均值: %+.3f  (n=%d 组)' % (np.mean(within_corr), len(within_corr)))
    print('  组内 Spearman > 0 的比例: %.1f%%' % (100*np.mean(np.array(within_corr) > 0)))

print()
print('=' * 76)
print('C) 分层：按 prefix_steps')
print('=' * 76)
for ps in sorted(set(r['ps'] for r in rows)):
    sub = [r for r in rows if r['ps'] == ps]
    print('  ps=%-3d n=%-3d  V mean=%+.4f  p mean=%.3f  |  r(V,p)=%.3f' % (
        ps, len(sub), np.mean([x['V'] for x in sub]), np.mean([x['p'] for x in sub]),
        pearson([x['V'] for x in sub], [x['p'] for x in sub])))

print()
print('=' * 76)
print('D) 控制 token length：在相近 ntok 内比较')
print('=' * 76)
ntoks = np.array([r['ntok'] for r in rows])
if len(rows) >= 6:
    qs = np.percentile(ntoks, [33, 67])
    for lo, hi, name in [(0, qs[0], 'short'), (qs[0], qs[1], 'mid'), (qs[1], 1e9, 'long')]:
        sub = [r for r in rows if lo <= r['ntok'] < hi]
        if len(sub) >= 4:
            print('  ntok %-6s n=%-3d  r(V,p)=%.3f' % (
                name, len(sub), pearson([x['V'] for x in sub], [x['p'] for x in sub])))

print()
print('=' * 76)
print('E) 分布概览')
print('=' * 76)
ps_vals = sorted(set(r['p'] for r in rows))
print('  成功率取值分布:', {round(v,2): sum(1 for r in rows if abs(r['p']-v) < 1e-9) for v in ps_vals})
print('  成功率全 0 的状态点: %d/%d' % (sum(1 for r in rows if r['p'] == 0), len(rows)))
print('  成功率全 1 的状态点: %d/%d' % (sum(1 for r in rows if r['p'] == 1), len(rows)))
print('  中间态(0<p<1): %d/%d' % (sum(1 for r in rows if 0 < r['p'] < 1), len(rows)))

# 保存
outp = OUTDIR / 'analysis_rows.jsonl'
with open(outp, 'w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print()
print('明细已保存:', outp)
