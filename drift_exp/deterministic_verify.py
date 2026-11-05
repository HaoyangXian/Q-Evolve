#!/usr/bin/env python3
"""
P0 最后一道确定性验证：
用 critic 实际使用的 tokenizer + 完全相同的过滤逻辑，
从 solver_A_ms512.jsonl 重新推导「哪 189 条被保留、原始行号是什么」，
并与 LCS 映射逐条对比。

不依赖 LCS 实现，可复现、可审计。
"""
import json, sys, hashlib
import numpy as np
import pathlib

ROOT = pathlib.Path('/data3/xhy/codes/work/paper_repos/Q-Evolve')
sys.path.insert(0, str(ROOT / 'QEvolve'))
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from eval_agent.data_utils import get_chat_template

MAX_LENGTH = 32768
MAX_STEPS = 512
MODEL_NAME = str(ROOT / 'runs/olympiad_qwen3_8b_iterative1024_small/sft_success')
DATA = ROOT / 'drift_exp/solver_A_ms512.jsonl'

roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

print('=' * 80)
print('确定性复现：完全按 qa_data_fast.preprocess_traj_dataset_qa_fast 的逻辑')
print('=' * 80)
print('model_name =', MODEL_NAME)
print('max_length =', MAX_LENGTH, ' max_steps =', MAX_STEPS)
print()

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tok.pad_token = tok.eos_token
tok.model_max_length = MAX_LENGTH
tmpl = get_chat_template(MODEL_NAME)

def tokenize_incremental(tok, tmpl, messages):
    full_ids, offsets, acc = [], [], 0
    for msg in messages:
        prefix, suffix = tmpl.role_prefix_and_suffix.get(msg['role'], ('', ''))
        piece = prefix + msg['content'] + suffix
        ids = tok(piece, add_special_tokens=False).input_ids
        full_ids.extend(ids)
        acc += len(ids)
        offsets.append(acc)
    return full_ids, offsets

rows = [json.loads(x) for x in open(DATA, encoding='utf-8') if x.strip()]
print('读入 solver_A 行数:', len(rows))

kept = []       # (solver_row, qid, ep_len, n_tokens, skip_reason)
skipped = []

for row_idx, item in enumerate(rows):
    conv = item.get('conversations')
    if not conv:
        skipped.append((row_idx, item.get('id'), 'no_conv'))
        continue
    n_msgs = len(conv)
    step_idxs = [i for i in range(2, n_msgs, 2) if i + 1 < n_msgs][:MAX_STEPS]
    if not step_idxs:
        skipped.append((row_idx, item.get('id'), 'no_steps'))
        continue

    messages = []
    bad = None
    try:
        for i, sent in enumerate(conv):
            role = roles[sent["from"]]
            expected = "user" if i % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError(f'role mismatch {i}')
            messages.append({"role": role, "content": sent["value"]})
    except (ValueError, KeyError) as e:
        skipped.append((row_idx, item.get('id'), f'role:{e}'))
        continue

    full_ids, offsets = tokenize_incremental(tok, tmpl, messages)
    next_end = [(offsets[i + 2] - 1) if (i + 2 < n_msgs) else (offsets[i + 1] - 1) for i in step_idxs]

    if any(x >= MAX_LENGTH for x in next_end):
        skipped.append((row_idx, item.get('id'), f'too_long(max_next_end={max(next_end)})'))
        continue

    kept.append({
        'solver_row': row_idx,
        'qid': item.get('id'),
        'ep_len': len(step_idxs),
        'n_tokens': next_end[-1] + 1,
        'max_next_end': max(next_end),
    })

print('保留:', len(kept), ' 跳过:', len(skipped))
print()

print('=' * 80)
print('被跳过的行（确定性推导）')
print('=' * 80)
for r, q, reason in skipped:
    print('  row %-3d qid=%-18s 原因: %s' % (r, q, reason))

print()
print('=' * 80)
print('与 LCS 映射逐条对比')
print('=' * 80)
lcs_path = ROOT / 'drift_exp/A_traj_to_solver_mapping.tsv'
lcs = []
with open(lcs_path, encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4:
            lcs.append({'traj_id': int(p[0]), 'solver_row': int(p[1]), 'qid': p[2], 'ep_len': int(p[3])})

print('LCS 映射条数:', len(lcs))
print('确定性推导条数:', len(kept))

# 按 traj_id 顺序比较
ok_rows = ok_qids = ok_eps = 0
mismatches = []
for i, k in enumerate(kept):
    l = lcs[i] if i < len(lcs) else None
    if l is None:
        mismatches.append((i, 'missing_lcs'))
        continue
    same_row = (k['solver_row'] == l['solver_row'])
    same_qid = (k['qid'] == l['qid'])
    same_ep = (k['ep_len'] == l['ep_len'])
    ok_rows += same_row
    ok_qids += same_qid
    ok_eps += same_ep
    if not (same_row and same_qid and same_ep):
        mismatches.append((i, k, l))

print()
print('solver_row 一致: %d/%d' % (ok_rows, len(kept)))
print('qid        一致: %d/%d' % (ok_qids, len(kept)))
print('ep_len     一致: %d/%d' % (ok_eps, len(kept)))
print()
if mismatches:
    print('不一致样例（前 10）:')
    for m in mismatches[:10]:
        print('  ', m)
else:
    print('★★★ 完全一致：确定性推导 == LCS 映射 (189/189) ★★★')
    print('→ A 档映射正式验证通过')

print()
print('=' * 80)
print('与已存档的 advantage 输出交叉验证（traj_id / ep_len）')
print('=' * 80)
adv = [json.loads(x) for x in open(ROOT / 'advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl') if x.strip()]
adv_by = {r['traj_id']: r for r in adv}
ok_adv = 0
for i, k in enumerate(kept):
    a = adv_by.get(i)
    if a is not None and a['ep_len'] == k['ep_len']:
        ok_adv += 1
print('推导 ep_len 与 advantage 输出一致: %d/%d' % (ok_adv, len(kept)))

# 输出确定性映射表
out = ROOT / 'drift_exp/A_mapping_deterministic.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('traj_id\tsolver_row\tqid\tep_len\tn_tokens\tmax_next_end\n')
    for i, k in enumerate(kept):
        f.write('%d\t%d\t%s\t%d\t%d\t%d\n' % (i, k['solver_row'], k['qid'], k['ep_len'], k['n_tokens'], k['max_next_end']))
print()
print('确定性映射已写入:', out)
