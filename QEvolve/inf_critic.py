#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import gc
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from peft import LoraConfig
from safetensors.torch import load_file as safe_load_file


def load_probe_full_state_dict(
    base_model_name_or_path: str,
    ckpt_dir: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 4096,
    strict: bool = True,
    use_done_head: bool = True,
):
    """
    适用于：Trainer 保存的整模型 checkpoint（full state_dict）
      - ckpt_dir/model.safetensors
      - 或 ckpt_dir/checkpoint-*/model.safetensors（自动取最新）
    要求：先实例化 ProbeModel，然后 load 参数
    """
    from train_critic import ProbeModel  # 确保是你训练用的同一个类

    # resolve checkpoint dir: pick latest checkpoint-* if needed
    if not os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
        if ckpts:
            cand = ckpts[-1]
            if os.path.exists(os.path.join(cand, "model.safetensors")):
                ckpt_dir = cand

    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.safetensors not found: {model_path}")

    # build same LoRA structure as training
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
    )

    # instantiate on CPU first to avoid VRAM spike
    model = ProbeModel(base_model_name_or_path, lora=lora, max_length=max_length, use_done_head=use_done_head)
    model.eval()

    # load state_dict (CPU)
    state_dict = safe_load_file(model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    del state_dict

    if missing or unexpected:
        print(f"[load_state_dict] strict={strict}")
        if missing:
            print("[load_state_dict] missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print("[load_state_dict] unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    # move to device + dtype
    model = model.to(device=device) #, dtype=dtype)
    model.eval()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("ProbeModel instance has no tokenizer. Check ProbeModel.__init__")

    return model, tokenizer, ckpt_dir

import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple


def _make_traj_meta(hf_ds, max_length: int) -> List[Dict[str, Any]]:
    """
    为每条 traj 预计算长度 L、valid_steps、positions、reward/done。
    只保存必要信息，避免在主循环里重复 Python list comprehension。
    """
    metas = []
    for traj_id in range(len(hf_ds)):
        item = hf_ds[traj_id]
        T = int(item.get("ep_len", 0))
        if T <= 0:
            continue

        input_ids = item["input_ids"][:max_length]
        L = len(input_ids)
        if L <= 1:
            continue

        # valid steps
        valid_steps = [
            t for t in range(T)
            if int(item["next_end"][t]) >= 0
            and int(item["next_end"][t]) < max_length
            and int(item["next_end"][t]) < L
        ]
        if not valid_steps:
            continue

        metas.append({
            "traj_id": traj_id,
            "L": L,
            "input_ids": input_ids,
            "valid_steps": valid_steps,
            "state_pos": [int(item["state_end"][t])  for t in valid_steps],
            "action_pos":[int(item["action_end"][t]) for t in valid_steps],
            "next_pos":  [int(item["next_end"][t])   for t in valid_steps],
            "reward":    [float(item["reward"][t])   for t in valid_steps],
            "done":      [float(item["done"][t])     for t in valid_steps],
            "ep_len":    int(item.get("ep_len", 0)),
            "final_done":int(item.get("final_done", 0)),
            "is_fail":   int(item.get("is_fail", 0)),
        })
    return metas


def _build_length_sorted_batches(
    metas: List[Dict[str, Any]],
    *,
    max_tokens: int,
    max_examples: int,
) -> List[List[int]]:
    """
    metas 已按 L 排序后，做 padding-aware batch:
      projected_tokens = (batch_size + 1) * max_seq_len_after_add
    """
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_maxL = 0

    for i, m in enumerate(metas):
        L = int(m["L"])
        new_maxL = max(cur_maxL, L)
        projected = (len(cur) + 1) * new_maxL
        if (len(cur) + 1 > max_examples) or (projected > max_tokens):
            if cur:
                batches.append(cur)
            cur = [i]
            cur_maxL = L
        else:
            cur.append(i)
            cur_maxL = new_maxL

    if cur:
        batches.append(cur)
    return batches


@torch.inference_mode()
def infer_all_by_traj_batched(
    model: torch.nn.Module,
    hf_ds,
    max_length: int,
    gamma: float = 0.95,
    out_jsonl: Optional[str] = None,
    *,
    max_tokens: int = 65536,     # 例如 H100 上可以很大；你也可先用 32768
    max_examples: int = 8,       # 防止 batch 太多条导致 Kmax 太大/CPU 组 batch 太慢
    show_batch_log: bool = False
) -> Dict[str, Any]:
    """
    Batched trajectory inference:
      - 先按 token length 排序
      - 再按 padding-aware token budget 组 batch
      - 每个 batch 两次 forward：Q / V
    """
    device = next(model.parameters()).device
    inner = model.module if hasattr(model, "module") else model
    inner.eval()

    pad_id = getattr(getattr(inner, "tokenizer", None), "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    # 1) build meta + sort by length
    metas = _make_traj_meta(hf_ds, max_length=max_length)
    # length sort, from long to short
    metas.sort(key=lambda x: x["L"], reverse=True)
    
    # reverse

    # 2) build batches
    batches = _build_length_sorted_batches(metas, max_tokens=max_tokens, max_examples=max_examples)

    if show_batch_log:
        lens = [m["L"] for m in metas]
        print(f"[Infer] num_trajs={len(metas)}, len_min={min(lens)}, len_max={max(lens)}, batches={len(batches)}")

    # fout = open(out_jsonl, "w", encoding="utf-8") if out_jsonl else None

    # global stats
    total_traj = 0
    total_steps = 0
    sum_Q = 0.0
    sum_V = 0.0
    sum_A = 0.0
    sum_td_err2 = 0.0
    res = []

    for b_ids in tqdm(batches, desc="Infer Q/V (batched, length-sorted)"):
        B = len(b_ids)
        batch_m = [metas[i] for i in b_ids]

        maxL = max(m["L"] for m in batch_m)
        Kmax = max(len(m["valid_steps"]) for m in batch_m)

        # ---- pad input_ids to [B, maxL] ----
        input_ids = torch.full((B, maxL), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, maxL), dtype=torch.long, device=device)

        # ---- pad positions to [B, Kmax] with -1 ----
        state_pos  = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
        action_pos = torch.full((B, Kmax), -1, dtype=torch.long, device=device)
        next_pos   = torch.full((B, Kmax), -1, dtype=torch.long, device=device)

        reward = torch.zeros((B, Kmax, 1), dtype=torch.float32, device=device)
        done   = torch.zeros((B, Kmax, 1), dtype=torch.float32, device=device)

        for i, m in enumerate(batch_m):
            ids = m["input_ids"]
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :L] = 1

            K = len(m["valid_steps"])
            state_pos[i, :K]  = torch.tensor(m["state_pos"], dtype=torch.long, device=device)
            action_pos[i, :K] = torch.tensor(m["action_pos"], dtype=torch.long, device=device)
            next_pos[i, :K]   = torch.tensor(m["next_pos"], dtype=torch.long, device=device)

            reward[i, :K, 0]  = torch.tensor(m["reward"], dtype=torch.float32, device=device)
            done[i, :K, 0]    = torch.tensor(m["done"], dtype=torch.float32, device=device)

        step_mask = (action_pos >= 0).float()  # [B,Kmax]  用来忽略 pad steps

        # ---- forward Q (一次) ----
        out_q = inner.forward_q_multi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_pos=action_pos,
        )
        Q1 = out_q["Q1"].float()  # [B,Kmax,1]
        Q2 = out_q["Q2"].float()
        Qmin = torch.minimum(Q1, Q2)

        # ---- forward V (一次) ----
        out_v = inner.forward_v_multi(
            input_ids=input_ids,
            attention_mask=attention_mask,
            state_pos=state_pos,
            next_pos=next_pos,
        )
        V  = out_v["V"].float()
        Vn = out_v["V_next"].float()

        td_target = reward + (1.0 - done) * gamma * Vn
        A = Qmin - V
        td_err = Qmin - td_target

        # ---- global stats (masked mean) ----
        denom = step_mask.sum().clamp_min(1.0).item()
        total_steps_batch = int(step_mask.sum().item())
        total_traj += B
        total_steps += total_steps_batch

        # 这里用 sum 而不是 mean*count，避免数值误差
        sum_Q += float((Qmin.squeeze(-1) * step_mask).sum().item())
        sum_V += float((V.squeeze(-1) * step_mask).sum().item())
        sum_A += float((A.squeeze(-1) * step_mask).sum().item())
        sum_td_err2 += float(((td_err.squeeze(-1).pow(2)) * step_mask).sum().item())

        # ---- optional write per-traj arrays ----
        Q1_cpu = Q1.squeeze(-1).detach().cpu()
        Q2_cpu = Q2.squeeze(-1).detach().cpu()
        Qm_cpu = Qmin.squeeze(-1).detach().cpu()
        V_cpu  = V.squeeze(-1).detach().cpu()
        Vn_cpu = Vn.squeeze(-1).detach().cpu()
        A_cpu  = A.squeeze(-1).detach().cpu()
        td_cpu = td_target.squeeze(-1).detach().cpu()
        te_cpu = td_err.squeeze(-1).detach().cpu()

        for i, m in enumerate(batch_m):
            K = len(m["valid_steps"])
            rec = {
                "traj_id": int(m["traj_id"]),
                "ep_len": int(m["ep_len"]),
                "final_done": int(m["final_done"]),
                "is_fail": int(m["is_fail"]),
                "valid_steps": m["valid_steps"],

                "Q1": Q1_cpu[i, :K].tolist(),
                "Q2": Q2_cpu[i, :K].tolist(),
                "Qmin": Qm_cpu[i, :K].tolist(),
                "V": V_cpu[i, :K].tolist(),
                "V_next": Vn_cpu[i, :K].tolist(),
                "A": A_cpu[i, :K].tolist(),
                "td_target": td_cpu[i, :K].tolist(),
                "td_err": te_cpu[i, :K].tolist(),
                "reward": m["reward"],
                "done": m["done"],
            }
            res.append(rec)
        with open(out_jsonl, "a", encoding="utf-8") as fout:
            for i in range(B):
                json_line = json.dumps(res[-B + i], ensure_ascii=False)
                fout.write(json_line + "\n")


    # 注意：sum_* 现在是 masked sum，所以 mean = sum / total_steps
    stats = {
        "num_trajs_inferred": int(total_traj),
        "total_steps": int(total_steps),
        "Qmin_mean": float(sum_Q / max(total_steps, 1)),
        "V_mean": float(sum_V / max(total_steps, 1)),
        "A_mean": float(sum_A / max(total_steps, 1)),
        "td_err_mse": float(sum_td_err2 / max(total_steps, 1)),
        "out_jsonl": out_jsonl,
        "batching": {
            "max_tokens": int(max_tokens),
            "max_examples": int(max_examples),
            "num_batches": int(len(batches)),
        }
    }
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/train/alfworld/merged_alfworld_q-iter1.jsonl', help="trajectory jsonl (same file used to train the critic)")
    parser.add_argument("--task", type=str, default="alfworld", choices=["alfworld", "sciworld", "webshop", "qa"])
    parser.add_argument("--use_aux_reward", dest="use_aux_reward", action="store_true", default=True)
    parser.add_argument("--no_aux_reward", dest="use_aux_reward", action="store_false")
    parser.add_argument("--failed_data_path", type=str, default=None)
    parser.add_argument("--advantage_base_model", type=str, required=True, help="base backbone model id/path")
    parser.add_argument("--advantage_model_path", type=str, required=True, help="dir containing model.safetensors (full state_dict)")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--exp_name", type=str, default="default")

    parser.add_argument("--strict", action="store_true", help="strict load_state_dict")
    # parser.add_argument("--no_done_head", action="store_true", help="set use_done_head=False when building model")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # dataset: build in memory from the same jsonl used to train the critic
    # (row order is deterministic, so traj_id indices align with train_policy)
    from train_critic import build_traj_dataset
    hf_ds = build_traj_dataset(
        task=args.task,
        data_path=args.data_path,
        model_name=args.advantage_base_model,
        failed_data_path=args.failed_data_path,
        out_dir=None,
        gamma=args.gamma,
        max_length=args.max_length,
        use_aux_reward=args.use_aux_reward,
    )
    print(f"[Build] trajectories = {len(hf_ds)} from {args.data_path}")

    # define save path: advantage_inference/<task>_traj_tokens/<w|wo>_assigned_rewards/<exp_name>/...
    sub = 'w_assigned_rewards' if args.use_aux_reward else 'wo_assigned_rewards'
    out_dir = os.path.join("advantage_inference", f"{args.task}_traj_tokens", sub, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    args.out_jsonl = os.path.join(out_dir, f"adv_token_pred-gamma{args.gamma}-len{args.max_length}-{args.exp_name}.jsonl")
    # infer_all_by_traj_batched appends; start from a clean file
    if os.path.exists(args.out_jsonl):
        os.remove(args.out_jsonl)
    print(f"[Setup] out_jsonl = {args.out_jsonl}")
    
    # model
    model, tokenizer, resolved = load_probe_full_state_dict(
        base_model_name_or_path=args.advantage_base_model,
        ckpt_dir=args.advantage_model_path,
        device=device,
        # dtype=dtype,
        max_length=args.max_length,
        strict=args.strict,
        use_done_head=False,  # do not use done head during inference
    )
    print(f"[Load]  Advantage model from {resolved}")

    # inference
    stats = infer_all_by_traj_batched(
        model=model,
        hf_ds=hf_ds,
        max_length=args.max_length,
        gamma=args.gamma,
        out_jsonl=args.out_jsonl,      # 可选
        # Patched: the H100-era batch cap (262144 tok / 128 trajs) OOMs on a 40G A100.
        max_tokens=int(os.environ.get("INF_MAX_TOKENS", 32768)),
        max_examples=int(os.environ.get("INF_MAX_EXAMPLES", 8)),
        show_batch_log=True
    )
    print(stats)

    print("[Inference stats]")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
