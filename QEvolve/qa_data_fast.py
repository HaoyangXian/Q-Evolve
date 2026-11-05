import json, os
from typing import Optional
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from eval_agent.data_utils import get_chat_template

def _qa_prompt(tokenizer, model_name, conversations):
    roles={"human":"user","gpt":"assistant","user":"user","assistant":"assistant"}
    messages=[]
    for i,sent in enumerate(conversations):
        role=roles[sent["from"]]; expected="user" if i%2==0 else "assistant"
        if role != expected: raise ValueError(f"role mismatch {i}: {role} != {expected}")
        messages.append({"role":role,"content":sent["value"]})
    return get_chat_template(model_name).get_prompt(messages)

def _tokenize_incremental(tok, tmpl, messages):
    '''逐消息增量 tokenize，返回 (full_ids, offsets)。
    offsets[k] = 第 k 个消息结束时的累计 token 数。
    与完整一次性 tokenize 完全一致（Qwen/LLaMA PLAIN 模板 prefix/suffix 固定）。'''
    full_ids = []
    offsets = []
    acc = 0
    for msg in messages:
        role, content = msg['role'], msg['content']
        prefix, suffix = tmpl.role_prefix_and_suffix.get(role, ('', ''))
        piece = prefix + content + suffix
        piece_ids = tok(piece, add_special_tokens=False).input_ids
        full_ids.extend(piece_ids)
        acc += len(piece_ids)
        offsets.append(acc)
    return full_ids, offsets

def preprocess_traj_dataset_qa_fast(data_path:str, failed_data_path:Optional[str], model_name:str, out_dir=None, gamma=0.95, max_steps=128, max_length=4096, use_aux_reward=False):
    rows=[json.loads(x) for x in open(data_path,encoding='utf-8') if x.strip()]
    if failed_data_path:
        rows += [json.loads(x) for x in open(failed_data_path,encoding='utf-8') if x.strip()]
    tok=AutoTokenizer.from_pretrained(model_name,use_fast=True); tok.pad_token=tok.eos_token; tok.model_max_length=max_length
    tmpl = get_chat_template(model_name)
    roles={"human":"user","gpt":"assistant","user":"user","assistant":"assistant"}
    records=[]
    for item in rows:
        conv=item.get('conversations')
        if not conv: continue
        n_msgs = len(conv)
        # 复现原逻辑的 step 边界：idx in range(2, n_msgs, 2)，且 idx+1 < n_msgs
        step_idxs = [idx for idx in range(2, n_msgs, 2) if idx+1 < n_msgs]
        step_idxs = step_idxs[:max_steps]
        if not step_idxs: continue
        n_steps = len(step_idxs)

        # 构建 messages（含 role 校验）
        messages=[]
        try:
            for i,sent in enumerate(conv):
                role=roles[sent["from"]]; expected="user" if i%2==0 else "assistant"
                if role != expected: raise ValueError(f"role mismatch {i}")
                messages.append({"role":role,"content":sent["value"]})
        except (ValueError, KeyError):
            continue

        # 增量 tokenize 整条轨迹（O(n) 而非 O(n^2)）
        full_ids, offsets = _tokenize_incremental(tok, tmpl, messages)

        # 每个 step 的 state/action/next 结束 token 位置（与原逻辑逐前缀 tokenize 完全等价）
        state_end  = [offsets[idx]   - 1 for idx in step_idxs]   # state  = conv[:idx+1]，最后索引 idx
        action_end = [offsets[idx+1] - 1 for idx in step_idxs]   # action = conv[:idx+2]，最后索引 idx+1
        next_end   = [(offsets[idx+2] - 1) if (idx+2 < n_msgs) else (offsets[idx+1] - 1) for idx in step_idxs]
        #   next    = conv[:idx+3] 若 idx+2<n_msgs，否则退化为 action

        # 超长跳过（与原逻辑一致：任一步 next 位置 >= max_length 则丢弃整条轨迹）
        if any(x >= max_length for x in next_end):
            continue

        rewards=item.get('assigned_reward') or [0.0]*n_steps
        rewards=[float(x) for x in rewards[:n_steps]]; rewards += [0.0]*(n_steps-len(rewards))
        meta=item.get('meta') or {}; final_reward=float(meta.get('reward',item.get('final_reward',0.0)) or 0.0)
        rewards[-1]+=final_reward

        ep=n_steps
        records.append({'input_ids':full_ids[:next_end[-1]+1],'state_end':state_end,'action_end':action_end,'next_end':next_end,'reward':rewards[:ep],'done':[0.0]*(ep-1)+[1.0],'weight':((np.arange(1,ep+1,dtype=np.float32)/ep + (1.0 if final_reward>0 else 0.0))*0.5+0.5).tolist(),'final_done':1,'ep_len':ep})
    ds=Dataset.from_list(records)
    if out_dir: os.makedirs(out_dir,exist_ok=True); ds.save_to_disk(out_dir)
    return ds
