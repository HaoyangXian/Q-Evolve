#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import functional as F

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model
from trl.models import create_reference_model
import numpy as np

# -------------------------
# Utils: weight transform
# -------------------------

def low_var_kl_penalty_from_logratio(log_ratio: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    """
    log_ratio = log(q/p) = logp_ref - logp_new
    returns: exp(log_ratio) - log_ratio - 1  (>=0), elementwise
    """
    log_ratio = torch.clamp(log_ratio, min=-clip, max=clip)
    return torch.exp(log_ratio) - log_ratio - 1.0



def build_traj_step_map_from_qv_jsonl(
    qv_jsonl: str,
    *,
    adv_weight_mode: str = "pos",
    topk_per_traj: int = 0,
    drop_zero_weight: bool = True,
    outcome_advantage: bool = False,
    precomputed_advantage: bool = False,

) -> Dict[int, List[Tuple[int, float]]]:
    """
    改版：不再直接使用 rec["A"]，而是用 GAE(lambda) 从 reward/done/V/V_next 估计 A。
    输出: traj_id -> [(step_t, A_gae_t), ...]
    """
    gamma = 0.95   # ✅ 写死，不加参数；想改就改这里
    lam   = 0.95   # ✅ 写死，不加参数；想改就改这里

    traj2steps: Dict[int, List[Tuple[int, float]]] = {}

    with open(qv_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            traj_id = int(rec["traj_id"])
            valid_steps = [int(t) for t in rec["valid_steps"]]
            # print(valid_steps)

            # ---- 必要字段 ----
            reward = rec.get("reward", None)
            done   = rec.get("done", None)
            
            V      = rec.get("V", None)
            Q      = [(q1 + q2) / 2 for q1, q2 in zip(rec['Q1'], rec['Q2'])]
            # state_end = rec.get("state_pos", None)
            # action_end = rec.get("action_pos", None)
            V_next = rec.get("V_next", None)
            if reward is None or done is None or V is None or V_next is None:
                raise ValueError(f"traj {traj_id} missing reward/done/V/V_next in qv_jsonl")

            # policy advantage uses ENV reward only — aux penalties (negative) shape the CRITIC (Q/V),
            # NOT the policy objective. Floor at 0 to strip aux: env reward is >=0 (binary 1 for alfworld
            # OR fractional score for sciworld -> kept), aux penalties are <0 -> zeroed. Env-agnostic.
            reward = [max(float(r), 0.0) for r in reward]

            T = len(reward)
            # 防御性检查
            if not (len(done) == T and len(V) == T and len(V_next) == T):
                raise ValueError(f"len mismatch traj {traj_id}: "
                                 f"reward={len(reward)} done={len(done)} V={len(V)} V_next={len(V_next)}")

            if precomputed_advantage:
                A_gae = [float(x) for x in rec.get("A", [])]
                if len(A_gae) != T:
                    raise ValueError(f"traj {traj_id} precomputed A length {len(A_gae)} != {T}")
            else:
                # ---- 1-step TD residual δ_t ----
                delta = [0.0] * T
                for t in range(T):
                    m = 1.0 - float(done[t])
                    td_target = float(reward[t]) + gamma * m * float(V_next[t])
                    delta[t] = td_target - float(V[t])
                # ---- GAE(λ): backward recursion ----
                A_gae = [0.0] * T
                gae = 0.0
                for t in range(T - 1, -1, -1):
                    m = 1.0 - float(done[t])
                    gae = delta[t] + gamma * lam * m * gae
                    A_gae[t] = gae
            
            # Sparse-reward QA can give positive TD errors to failed prefixes
            # when the critic baseline is below zero.  Use the terminal
            # outcome as a signed offline preference signal when requested.
            if outcome_advantage:
                outcome = 1.0 if float(reward[-1]) > 0.0 else -1.0
                A_gae = [outcome] * T

            # ---- build items for valid_steps ----
            items = {}
                
            for t in valid_steps:
                if t < 0 or t >= T:
                    continue
                a = float(A_gae[t])
                items[t] = a

            # QA prefix-MDPs may contain a single terminal action.  Keeping
            # one-step trajectories is also useful for the contextual-bandit
            # ablation; the old >1 filter silently discarded them.
            if len(items) > 0:
                traj2steps[traj_id] = items

    return traj2steps

# -------------------------
# Dataset: each sample = one trajectory
# -------------------------
class TrajectoryMultiStepDataset(Dataset):
    """
    每条样本 = 一个 trajectory
    collator 会在该 trajectory 内采样 K 个 step，展开成 K 条 LM 样本做一次优化。
    """
    def __init__(
        self,
        hf_ds,
        traj2steps: Dict[int, List[Tuple[int, float]]],
        *,
        max_length: int = 4096,
    ):
        self.hf_ds = hf_ds
        self.traj2steps = traj2steps
        self.max_length = int(max_length)

        self.traj_ids: List[int] = sorted(list(traj2steps.keys()))

        # 给 dynamic batching 预估 token cost：K * (该 traj 内可用 step 的最大 action_end)
        self.lengths: List[int] = []
        for traj_id in self.traj_ids:
            item = self.hf_ds[traj_id]
            # 估计 padding-aware cost（该 traj 内 K 个 step 会 pad 到 max_len）
            max_len = int(item['action_end'][-1]) + 1
            self.lengths.append(max_len)

    def __len__(self):
        return len(self.traj_ids)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        traj_id = int(self.traj_ids[i])
        return {"traj_id": traj_id}


# -------------------------
# Dynamic token budget batch sampler (padding-aware + DDP-safe)
# -------------------------
class TokenBudgetBatchSampler(Sampler[List[int]]):
    """
    动态 batch：每个 batch 的总 token 数（padding-aware）<= max_tokens
    DDP-safe：先构造全局 batches，再 pad 到 world_size 倍数，然后每个 rank 取切片。
    """
    def __init__(
        self,
        lengths: List[int],
        max_tokens: int,
        max_examples: int = 999999,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        sort_by_length: bool = True,
        bucket_size: int = 256,
    ):
        self.lengths = lengths
        self.max_tokens = int(max_tokens)
        self.max_examples = int(max_examples)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last
        self.sort_by_length = sort_by_length
        self.bucket_size = int(bucket_size)

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.epoch = 0
        self._batches_rank: List[List[int]] = []

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)  # ✅ 修复你原来的 typo
        self._build_batches()

    def _build_batches(self):
        n = len(self.lengths)
        indices = list(range(n))

        rng = random.Random(self.seed + self.epoch * 1000003)
        if self.shuffle:
            rng.shuffle(indices)

        # bucket 内按长度排序，减少 padding
        if self.sort_by_length and self.bucket_size > 0:
            buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
            new_indices = []
            for b in buckets:
                b.sort(key=lambda i: self.lengths[i])
                new_indices.extend(b)
            indices = new_indices
        elif self.sort_by_length:
            indices.sort(key=lambda i: self.lengths[i])

        global_batches: List[List[int]] = []
        cur: List[int] = []
        cur_max = 0

        for idx in indices:
            L = int(self.lengths[idx])
            if L <= 0:
                continue

            if L > self.max_tokens:
                if cur:
                    global_batches.append(cur)
                global_batches.append([idx])
                cur = []
                cur_max = 0
                continue

            new_max = max(cur_max, L)
            projected = (len(cur) + 1) * new_max  # padding-aware（近似）

            if projected > self.max_tokens or (len(cur) + 1 > self.max_examples):
                if cur:
                    global_batches.append(cur)
                cur = [idx]
                cur_max = L
            else:
                cur.append(idx)
                cur_max = new_max

        if cur:
            global_batches.append(cur)

        if self.drop_last and len(global_batches) % self.world_size != 0:
            global_batches = global_batches[: (len(global_batches) // self.world_size) * self.world_size]

        if len(global_batches) == 0:
            global_batches = [[0]]

        while len(global_batches) % self.world_size != 0:
            global_batches.append(global_batches[-1])

        self._batches_rank = global_batches[self.rank :: self.world_size]

    def __iter__(self):
        if not self._batches_rank:
            self._build_batches()
        for b in self._batches_rank:
            yield b

    def __len__(self):
        if not self._batches_rank:
            self._build_batches()
        return len(self._batches_rank)


# -------------------------
# Collator: sample K steps per traj, then flatten
# -------------------------
@dataclass
class TrajAllStepMaskCollator:
    tokenizer: PreTrainedTokenizerBase
    hf_ds: Any
    traj2steps: Dict[int, List[Tuple[int, float]]]
    max_length: int = 4096
    debug_batch: bool = False

    def set_epoch(self, epoch: int):
        pass  # 不采样了

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        traj_input_ids: List[List[int]] = []
        traj_attn: List[List[int]] = []
        step_starts: List[List[int]] = []
        think_ends: List[List[int]] = []
        step_ends: List[List[int]] = []
        step_weights: List[List[float]] = []
        think_masks: List[List[int]] = []

        for f in features:
            traj_id = int(f["traj_id"])
            item = self.hf_ds[traj_id]

            steps = self.traj2steps[traj_id]  # ✅ 全部 step
            valid_steps = item["valid_step_idx"]
            # 过滤掉无效 step（non-Thought/Action）

            if len(steps) == 0:
                # 理论上不会发生（traj2steps 构建时已过滤）
                steps = [(0, 0.0)]

            # 用所有 step 的最大 action_end 决定这条 trajectory 截断长度
            max_ae = 0
            metas = []
            # for (t, w) in steps:
            for iii in range(len(item["state_end"])):
                if not (iii in valid_steps):
                    continue
                if not (iii in steps):
                    continue
                w = steps[iii]
                t = iii
                if w < -1.0:
                    continue
                assert iii == t
                state_end = int(item["state_end"][t])
                action_end = int(item["action_end"][t])
                think_end = int(item["think_end"][t])

                action_end = min(action_end, self.max_length - 1)
                if state_end >= action_end:
                    state_end = max(0, action_end - 1)

                max_ae = max(max_ae, action_end)
                metas.append((state_end, action_end, think_end, float(w)))

            ids = item["input_ids"][: max_ae + 1]  # ✅ 整段（到最后一个 action_end）
            think_mask = item["think_mask"][: max_ae + 1]
            traj_input_ids.append(ids)
            traj_attn.append([1] * len(ids))

            # shift-space span: [state_end, action_end-1]
            ss, ee, think_ee, ww = [], [], [], []
            for (state_end, action_end, think_end, w) in metas:
                s = state_end
                think_e = think_end - 1
                e = action_end - 1
                if e < s or think_e < s:
                    # s, e = -1, -1
                    raise RuntimeError(f"Invalid step span in traj {traj_id}: s={s}, e={e}, think_e={think_e}")
                    
                
                ss.append(int(s))
                ee.append(int(e))
                think_ee.append(int(think_e))
                ww.append(float(w))

            step_starts.append(ss)
            think_ends.append(think_ee)
            step_ends.append(ee)
            step_weights.append(ww)
            think_masks.append(think_mask)

        # pad K to K_max within this batch
        K_max = max(len(x) for x in step_starts)

        def pad_int(x, pad=-1):
            return x + [pad] * (K_max - len(x))

        def pad_float(x, pad=0.0):
            return x + [pad] * (K_max - len(x))

        step_starts = [pad_int(x) for x in step_starts]
        think_ends = [pad_int(x) for x in think_ends]
        step_ends = [pad_int(x) for x in step_ends]
        step_weights = [pad_float(x) for x in step_weights]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [{"input_ids": ids, "attention_mask": attn} for ids, attn in zip(traj_input_ids, traj_attn)],
            padding=True,
            return_tensors="pt",
        )

        batch["step_starts"] = torch.tensor(step_starts, dtype=torch.long)
        batch["think_ends"] = torch.tensor(think_ends, dtype=torch.long)
        batch["step_ends"] = torch.tensor(step_ends, dtype=torch.long)
        batch["step_weights"] = torch.tensor(step_weights, dtype=torch.float32)
        
        # think_masks
        batch["think_mask"] = torch.zeros_like(batch["input_ids"])
        for i, f in enumerate(features):
            # batch["think_masks"][i, :] = torch.tensor(think_masks[i], dtype=torch.long)
            batch["think_mask"][i, :len(think_masks[i])] = torch.tensor(think_masks[i], dtype=torch.long)

        if self.debug_batch:
            bs = batch["input_ids"].size(0)
            max_len = batch["input_ids"].size(1)
            print(f"[Batch] traj_bs={bs} max_len={max_len} K_max={K_max}")

        return batch


def _outer_fsdp_module(module: Optional[nn.Module]) -> Optional[nn.Module]:
    """First FSDP-wrapped submodule (depth-first), or None if model is not under FSDP yet."""
    if module is None:
        return None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return None
    if isinstance(module, FSDP):
        return module
    for child in module.children():
        found = _outer_fsdp_module(child)
        if found is not None:
            return found
    return None


# -------------------------
# Trainer: weighted token CE + dynamic batching
# -------------------------
class WeightedSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        max_tokens_per_device: int = 16384,
        max_examples_per_device: int = 16,
        sort_by_length: bool = True,
        bucket_size: int = 256,
        adv_beta: float = 1.0,
        neg_mode=None,
        neg_tau=0.0, alpha_ul=0.1, 
        kl_coef=0.1,
        think_token_weight=1.0,
        eps_low=0.4,
        eps_high=0.4,
        ref_sync_steps=0,
        no_reference_model=False,
        task="generic",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_tokens_per_device = int(max_tokens_per_device)
        self.max_examples_per_device = int(max_examples_per_device)
        self.sort_by_length = bool(sort_by_length)
        self.bucket_size = int(bucket_size)
        self.adv_beta = float(adv_beta)
        self.neg_mode=neg_mode
        self.neg_tau = float(neg_tau)
        self.alpha_ul = float(alpha_ul)
        self.task = str(task)
        self.think_token_weight = float(think_token_weight)
        
        self.eps_low = eps_low
        self.eps_high = eps_high
        # 0 = ref frozen at the SFT init (KL anchor AND the BPPO ratio's behavior policy
        # both stay pinned to SFT); N>0 = old behavior, re-copy live policy into ref every
        # N global steps — with that, KL only bounds drift *since the last sync*, so
        # cumulative drift from SFT is unconstrained (observed to end in repetition collapse).
        self.ref_sync_steps = int(ref_sync_steps)

        # NEW:
        # if neg_mode == "unlikelihood":
        self.use_clip = not bool(no_reference_model)
        self.ref_model = None if no_reference_model else create_reference_model(self.model)
        if self.ref_model is not None:
            self._sync_ref_model()
        # self.ref_model = create_reference_model("meta-llama/Llama-2-7b-chat-hf")
        self.kl_coef = float(kl_coef)


        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            # 确保 ref 在同设备（Trainer 里一般用这个）
            try:
                self._move_model_to_device(self.ref_model, self.args.device)
            except Exception:
                pass

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        lengths = getattr(train_dataset, "lengths", None)
        if lengths is None:
            lengths = [1 for _ in range(len(train_dataset))]

        batch_sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            max_tokens=self.max_tokens_per_device,
            max_examples=self.max_examples_per_device,
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            sort_by_length=self.sort_by_length,
            bucket_size=self.bucket_size,
        )

        epoch = int(self.state.epoch or 0)
        batch_sampler.set_epoch(epoch)

        # ✅ 让 collator 的 step 采样随 epoch 变化
        if hasattr(self.data_collator, "set_epoch"):
            self.data_collator.set_epoch(epoch)

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        use_clip = bool(getattr(self, "use_clip", True))

        step_starts = inputs.pop("step_starts")   # [B, K] shift-space (inclusive)
        think_mask  = inputs.pop("think_mask")    # [B, T] or [B, ?] (must align with input_ids)
        step_ends   = inputs.pop("step_ends")     # [B, K] shift-space (inclusive)
        A_raw       = inputs.pop("step_weights")  # [B, K]
        
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # shift
        shift_logits = logits[:, :-1, :].contiguous()             # [B, S, V]
        shift_labels = inputs["input_ids"][:, 1:].contiguous()    # [B, S]
        shift_attn   = inputs["attention_mask"][:, 1:].contiguous()  # [B, S]

        shift_labels = shift_labels.masked_fill(shift_attn == 0, -100)

        B, S = shift_labels.shape
        V = shift_logits.size(-1)
        device = shift_logits.device

        # -------- token NLL (chunked) --------
        flat_logits = shift_logits.reshape(-1, V)     # [B*S, V]
        flat_labels = shift_labels.reshape(-1)        # [B*S]

        chunk = 4096
        loss_chunks = []
        for i in range(0, flat_labels.numel(), chunk):
            l = F.cross_entropy(
                flat_logits[i:i+chunk],
                flat_labels[i:i+chunk],
                ignore_index=-100,
                reduction="none",
            )
            loss_chunks.append(l)
        token_nll_new = torch.cat(loss_chunks, dim=0).view(B, S)  # [B,S]

        tok_valid = (shift_labels != -100).to(token_nll_new.dtype)  # [B,S]

        # ----- THINK token weight (用于 new/ref 的 logp 聚合 & step NLL 聚合) -----
        think_mask = think_mask.to(device=device)
        if think_mask.dtype != torch.bool:
            think_mask = think_mask > 0
        think_mask = think_mask[:, 1:]  # shift-space
        think_mask = think_mask & (shift_labels != -100)

        think_w = float(getattr(self, "think_token_weight", 1.0))
        token_w = torch.ones_like(token_nll_new)  # 不含 tok_valid
        token_w = torch.where(think_mask, token_w * think_w, token_w)

        # 最终用于“求和”的权重：mask padding + think weight
        w_sum = token_w * tok_valid  # [B,S]

        # ------------------------------
        # ref token NLL（clip 需要；KL anchor 也可复用）
        # ------------------------------
        kl_loss = token_nll_new.new_tensor(0.0)
        ref_token_nll = None
        resp_mask = torch.zeros_like(token_nll_new, dtype=torch.bool)
        kl_tok = torch.zeros_like(token_nll_new)
        need_ref = use_clip or (getattr(self, "ref_model", None) is not None and float(getattr(self, "kl_coef", 0.0)) > 0.0)
        if need_ref:
            if getattr(self, "ref_model", None) is None:
                raise RuntimeError("use_clip=True but self.ref_model is None. Provide old/ref model or store old logp in data.")
            with torch.no_grad():
                ref_out = self.ref_model(**inputs)
                ref_shift_logits = ref_out.logits[:, :-1, :].contiguous()

            ref_flat_logits = ref_shift_logits.reshape(-1, V)
            ref_chunks = []
            for i in range(0, flat_labels.numel(), chunk):
                rl = F.cross_entropy(
                    ref_flat_logits[i:i+chunk],
                    flat_labels[i:i+chunk],
                    ignore_index=-100,
                    reduction="none",
                )
                ref_chunks.append(rl)
            ref_token_nll = torch.cat(ref_chunks, dim=0).view(B, S)  # [B,S]

        # ------------------------------
        # 可选 KL anchor（保持你原来的 MSE(log_ratio)）
        # 注意：log_ratio 也应该用同样的 token mask（pad）；
        # 你原来用 union span mask，我保持不动，但修 device
        # ------------------------------
        if (getattr(self, "ref_model", None) is not None) and (float(getattr(self, "kl_coef", 0.0)) > 0.0):
            s = step_starts.to(device=device).long()
            e = step_ends.to(device=device).long()
            valid_span = (s >= 0) & (e >= 0) & (e >= s)

            # log_ratio = ref_nll - new_nll
            log_ratio = (ref_token_nll - token_nll_new)  # [B,S]

            s0 = s.clamp(min=0, max=S)
            e1 = (e + 1).clamp(min=0, max=S)

            diff = torch.zeros(B, S + 1, device=device, dtype=torch.float32)
            diff.scatter_add_(1, s0, valid_span.float())
            diff.scatter_add_(1, e1, -valid_span.float())
            resp_mask = (diff.cumsum(dim=1)[:, :-1] > 0)  # [B,S] bool

            resp_mask = resp_mask & (shift_labels != -100)


            # ✅ low-variance KL penalty: exp(log_ratio) - log_ratio - 1  (>=0)
            kl_clip = float(getattr(self, "kl_clip", 10.0))  # 可选：不给 args 就默认 10
            kl_tok = low_var_kl_penalty_from_logratio(log_ratio, clip=kl_clip)  # [B,S], >=0

            kl_tok = kl_tok * resp_mask.to(kl_tok.dtype)
            denom = resp_mask.to(kl_tok.dtype).sum(dim=1).clamp_min(1.0)
            kl_loss = (kl_tok.sum(dim=1) / denom).mean()

        # Low-memory QA smoke mode has no frozen reference model.  Reusing the
        # current policy log-probability gives ratio=1 and disables KL while
        # retaining the same clipped-policy code path.
        if ref_token_nll is None:
            ref_token_nll = token_nll_new.detach()

        # ============================================================
        # 1) 计算 step_logp_new / step_logp_ref（重点：两边都乘 think weight）
        # ============================================================
        s = step_starts.to(device=device).long()
        e = step_ends.to(device=device).long()
        A = A_raw.to(device=device).float()

        # A = torch.where(A > 0, 2 * A, A)   # 先对正的放大
        # A = torch.where(A > 0, torch.exp(A), A)   # 先对正的放大
        # A = torch.exp(A)
        
        valid_span = (s >= 0) & (e >= 0) & (e >= s)

        e_idx = e.clamp(min=0, max=S - 1)
        s_idx = (s - 1).clamp(min=0, max=S - 1)
        s_is0 = (s <= 0)

        # ✅ 这里是你要的：reference weight “乘上 think weight”
        # 用 w_sum（包含 think_w 与 pad mask）对 token logp 做加权求和
        logp_new_tok = (-token_nll_new) * w_sum
        logp_ref_tok = (-ref_token_nll) * w_sum

        pref_new = torch.cumsum(logp_new_tok, dim=1)
        pref_ref = torch.cumsum(logp_ref_tok, dim=1)

        new_e = pref_new.gather(1, e_idx); new_s = pref_new.gather(1, s_idx)
        ref_e = pref_ref.gather(1, e_idx); ref_s = pref_ref.gather(1, s_idx)

        new_s = torch.where(s_is0, torch.zeros_like(new_s), new_s)
        ref_s = torch.where(s_is0, torch.zeros_like(ref_s), ref_s)

        step_logp_new = torch.where(valid_span, new_e - new_s, torch.zeros_like(new_e))  # [B,K]
        step_logp_ref = torch.where(valid_span, ref_e - ref_s, torch.zeros_like(ref_e))  # [B,K]

        # ============================================================
        # 2) PPO clipped importance objective（不区分正负，但要按 A 符号选 min/max）
        # ============================================================

        # FIX(stability): ±20 allowed ratio up to e^20≈4.9e8. For A<0 the PPO min() branch is
        # UNBOUNDED in ratio (min picks ratio*A), so a drifted policy (ref only re-syncs every
        # 200 steps) explodes the loss — observed ratio_mean up to 2.7e6 and degenerate policies
        # on all three envs. ±2 bounds ratio to [0.135, 7.39], keeping the worst-case step term
        # ≤ ~7.4|A| while leaving the clip objective untouched near ratio≈1.
        delta = torch.clamp(step_logp_new - step_logp_ref, -2.0, 2.0)
        ratio = torch.exp(delta)

        # FIX(stability): bound the advantage magnitude (negatives are already floored at -0.5 by
        # the mask below; this caps the positive tail) so no single step dominates the gradient.
        A = A.clamp(-2.0, 2.0)

        surr1 = ratio * A
        
        eps_low = self.eps_low
        eps_high = self.eps_high

        ratio_clip = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
        surr2 = ratio_clip * A

        obj = torch.minimum(surr1, surr2)

        if self.neg_mode in ("drop", "unlikelihood"):
            # Positive-only policy improvement.  The old hard-coded A>=-0.5
            # condition accidentally trained on mildly negative QA prefixes.
            mask = valid_span & (A > max(0.0, self.neg_tau))
        else:
            mask = valid_span & (A != 0)
        mask_f = mask.to(obj.dtype)

        # per-traj loss: 每条轨迹内按 step 平均，再 batch 平均
        per_traj = -(obj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        loss_pg = per_traj.mean()

        loss_ul = loss_pg.new_tensor(0.0)
        if self.neg_mode == "unlikelihood":
            neg_mask = valid_span & (A < -max(0.0, self.neg_tau))
            neg_w = (-A).clamp(max=2.0) * neg_mask.to(obj.dtype)
            # log p(action) is negative; minimizing it lowers the probability
            # of prefixes carrying a negative advantage.
            loss_ul = (step_logp_new * neg_w).sum(dim=1) / neg_w.sum(dim=1).clamp_min(1.0)
            loss_ul = loss_ul.mean()

        loss = loss_pg + float(self.alpha_ul) * loss_ul + float(getattr(self, "kl_coef", 0.0)) * kl_loss

        # log
        self.dict_to_log = {
            "loss": float(loss.item()),
            "pg_loss": float(loss_pg.item()),
            "ul_loss": float(loss_ul.item()),
            "kl_loss": float(kl_loss.item()),
            "kl_coef": float(getattr(self, "kl_coef", 0.0)),
            "eps_clip_low": float(eps_low) if 'eps_low' in locals() else -1,
            "eps_clip_high": float(eps_high) if 'eps_high' in locals() else -1,
            "think_token_weight": float(think_w),
            "steps_used": float(mask.sum().item()),
            "A_mean": float((A * mask_f).sum().item() / mask_f.sum().clamp_min(1.0).item()),
            "ratio_mean": float((ratio * mask_f).sum().item() / mask_f.sum().clamp_min(1.0).item()),
            "log_ratio_mean": float((ref_token_nll - token_nll_new).masked_fill(~resp_mask, 0).sum().item() / resp_mask.sum().clamp_min(1).item()),
            "kl_tok_mean": float((kl_tok.sum().item() / resp_mask.sum().clamp_min(1).item())),

        }
        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )


        self.log(self.dict_to_log, start_time)

        if self.ref_sync_steps > 0 and self.state.global_step % self.ref_sync_steps == 0:
            self._sync_ref_model()

    def _sync_ref_model(self):
        
        if getattr(self, "ref_model", None) is None:
            return

        self.ref_model.eval()
        with torch.no_grad():
            model = getattr(self, "model", None)
            ref = getattr(self, "ref_model", None)
            if hasattr(self, "accelerator"):
                ref_u = self.accelerator.unwrap_model(ref)
            else:
                ref_u = ref

            # FSDP: unwrap_model().state_dict() 仍是分片/本地视图。注意 is_fsdp_enabled 在 __init__ 里就为 True，
            # 但此时模型尚未 prepare/FSDP 包装；对非 FSDP 模块用 FSDP.state_dict_type 会在 __exit__ 报 KeyError: None。
            policy_sd = None
            fsdp_root = _outer_fsdp_module(model)
            if fsdp_root is not None and hasattr(self, "accelerator"):
                if getattr(self.accelerator, "is_fsdp2", False):
                    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

                    opts = StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=True,
                        cpu_offload=False,
                    )
                    policy_sd = get_model_state_dict(model, options=opts)
                else:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

                    full_cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
                    try:
                        with FSDP.state_dict_type(fsdp_root, StateDictType.FULL_STATE_DICT, full_cfg):
                            policy_sd = fsdp_root.state_dict()
                    except KeyError:
                        # Some torch versions can raise on context exit when previous
                        # state_dict_type is unset (None). Use the newer checkpoint API.
                        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

                        opts = StateDictOptions(
                            full_state_dict=True,
                            broadcast_from_rank0=True,
                            cpu_offload=False,
                        )
                        policy_sd = get_model_state_dict(model, options=opts)

            if policy_sd is None:
                if hasattr(self, "accelerator"):
                    model_u = self.accelerator.unwrap_model(model)
                else:
                    model_u = model
                policy_sd = model_u.state_dict()

            ref_u.load_state_dict(policy_sd, strict=False)

        for p in self.ref_model.parameters():
            p.requires_grad_(False)
    # def _sync_ref_model(self):
    #     pass
# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--qv_jsonl", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="trajectory jsonl (same file used to train the critic / compute advantage)")
    parser.add_argument("--task", type=str, default="alfworld", choices=["alfworld", "sciworld", "webshop", "qa"])
    parser.add_argument("--use_aux_reward", dest="use_aux_reward", action="store_true", default=True)
    parser.add_argument("--no_aux_reward", dest="use_aux_reward", action="store_false")
    parser.add_argument("--failed_data_path", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--policy_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="policy_extracted")
    parser.add_argument("--max_length", type=int, default=4096)

    # advantage -> weight
    parser.add_argument("--adv_weight_mode", type=str, default="no", choices=["no", "pos", "sigmoid", "exp", "abs"])
    parser.add_argument("--adv_beta", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--topk_per_traj", type=int, default=0)
    parser.add_argument("--drop_zero_weight", action="store_true")
    parser.add_argument("--qa_outcome_advantage", action="store_true",
                        help="use solved/failed trajectory outcomes as signed QA advantages")
    parser.add_argument("--precomputed_advantage", action="store_true",
                        help="use rec[\"A\"] directly; for chunked full-trajectory policy")
    parser.add_argument("--think_token_weight", type=float, default=1.0)
    parser.add_argument("--eps_low", type=float, default=0.4)
    parser.add_argument("--eps_high", type=float, default=0.4)

    # ✅ NEW: 每条 trajectory 采样多少个 step
    # parser.add_argument("--steps_per_traj", type=int, default=4)

    # dynamic batching
    parser.add_argument("--max_tokens_per_device", type=int, default=22936)
    parser.add_argument("--max_examples_per_device", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--bucket_size", type=int, default=256)
    parser.add_argument("--no_length_sort", action="store_true")

    # memory savers
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--debug_batch", action="store_true")

    # train
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--ref_sync_steps", type=int, default=0,
                        help="0 (default) = ref_model frozen at the SFT checkpoint; "
                             "N>0 = re-sync ref from the live policy every N steps (legacy behavior: 200)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument(
        "--neg_mode",
        type=str,
        default="unlikelihood",
        choices=["drop", "unlikelihood", "all"],
        help="How to handle negative advantage steps: "
            "all=use all nonzero GAE steps; drop=only learn A>0; ul=drop + unlikelihood on A<0"
    )
    parser.add_argument("--neg_tau", type=float, default=0.0, help="threshold for negative: A < -neg_tau")
    parser.add_argument("--alpha_ul", type=float, default=0.1, help="weight for unlikelihood loss")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_reference_model", action="store_true",
                        help="disable the frozen reference copy; use with --kl_coef 0 for a low-memory QA smoke run")

    # ✅ FIX: args.bf16 你原来没定义

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Pin each rank to its own GPU early (prevents the multi-GPU "devices=0" CUDA init crash under torchrun)
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if _local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(_local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True

    # 3) tokenizer + model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) build trajectory dataset in memory from the same jsonl used to train the
    #    critic / compute advantage. Row order is deterministic, so traj_id indices
    #    in --qv_jsonl line up with the rows here (built with the same base tokenizer).
    from train_critic import build_traj_dataset
    hf_ds = build_traj_dataset(
        task=args.task,
        data_path=args.data_path,
        model_name=args.policy_model,
        failed_data_path=args.failed_data_path,
        out_dir=None,
        gamma=args.gamma,
        max_length=args.max_length,
        use_aux_reward=args.use_aux_reward,
    )
    print(f"[Build] trajs: {len(hf_ds)} from {args.data_path}")
    # add think ends
    # 常见： "Action:" 分隔 think 与 action

    def find_action_marker_idx(span_ids, tokenizer, lookahead=3):
        """
        在 span_ids 里找 'Action:' marker 的起始 token 位置（span 内 index）。
        返回: idx (>=0) 或 -1
        """
        toks = tokenizer.convert_ids_to_tokens(span_ids)

        def norm(t):
            # SentencePiece 的 '▁' 表示词首空格；去掉后比较更稳
            return t.replace("▁", "")
        for i, tk in enumerate(toks):
            t = norm(tk)

            # 1) token 自带 Action:（最稳）
            if "Action:" in t:
                return i

            # 2) token 是 Action（或包含 Action），后面跟冒号
            if t == "Action" or t.endswith("Action"):
                # 向后看几个 token 找 ':'
                for j in range(i + 1, min(len(toks), i + 1 + lookahead)):
                    if norm(toks[j]) == ":":
                        return i

        return -1


    def add_think_end(example):
        ids = example["input_ids"]
        state_ends = example["state_end"]
        action_ends = example["action_end"]

        think_ends = []
        think_masks = [0 for _ in range(len(ids))]  # default all think

        for state_end, action_end in zip(state_ends, action_ends):
            if state_end == state_ends[-1]:
                if state_end == action_end:
                    think_ends.append(action_end)
                    # break
                    
                    continue
                
            if state_end >= action_end:
                # it is possible that state_end == action_end if no next state
                raise RuntimeError(f"Invalid span: state_end {state_end} >= action_end {action_end}")

            action_start = state_end + 1
            action_end = min(action_end, len(ids) - 1)
            if action_start > action_end:
                raise RuntimeError(f"Invalid action span: start {action_start} > end {action_end}")

            span_ids = ids[action_start:action_end + 1]

            marker_i = find_action_marker_idx(span_ids, tokenizer)

            if marker_i == -1: # 0 or -1
                if args.task == "qa":
                    # QA actions do not contain ALFWorld's Thought/Action
                    # marker.  Treat the whole generated step as trainable
                    # policy text and leave think_mask at zero.
                    think_ends.append(action_start + 1)
                else:
                    think_ends.append(action_end)  # whole action span is think
                    think_masks[state_end+1:action_end+1] = [1]*(action_end - state_end)
                # break
                continue
            elif marker_i == 0:
                think_ends.append(action_start)  # think is empty
                # break
                
                continue
            think_end = action_start + marker_i - 1  # original-space, inclusive
            if think_end < action_start:
                raise RuntimeError(f"Invalid action span: start {action_start} > think end {think_end}")

            think_ends.append(think_end)
            think_masks[state_end+1:think_end+1] = [1]*(think_end - state_end)
        drop_non = 0
        valid_step = []
        for i in range(len(think_ends) - 1, -1, -1):
            te = think_ends[i]
            se = state_ends[i]
            ae = action_ends[i]
            step_txt = tokenizer.decode(ids[se+1:ae+1], skip_special_tokens=True)
            non_qa_format_bad = (not 'Thought:' in step_txt and not 'Action:' in step_txt)
            if ((ae - te) <= 1 or (ae - se) <= 1 or (te - se) <= 1
                    or (args.task != "qa" and non_qa_format_bad)):
                think_masks[se+1:ae+1] = [0]*(ae - se)
                # think_ends.pop(i)
                # state_ends.pop(i)
                # action_ends.pop(i)
                drop_non += 1
                continue
            valid_step.append(i)
        valid_step = valid_step[::-1]

            
        example["num_steps_dropped_non"] = drop_non
        example["think_end"] = think_ends
        example["valid_step_idx"] = valid_step
        # concat
    
        example["state_end"] = state_ends
        example["action_end"] = action_ends
        example["input_ids"] = example["input_ids"]
        example["think_mask"] = think_masks
        assert len(think_ends) == len(state_ends) and len(example["think_mask"]) == len(example["input_ids"])
        return example

    # ===== 3) map 添加列 =====
    hf_ds = hf_ds.map(add_think_end)
    
    # print(f"Dropped steps stats: mean={np.mean(hf_ds['num_steps_dropped']):.2f}, max={np.max(hf_ds['num_steps_dropped'])}, total dropped={np.sum(hf_ds['num_steps_dropped'])}")
    print(f"Dropped non-informative steps stats: mean={np.mean(hf_ds['num_steps_dropped_non']):.2f}, max={np.max(hf_ds['num_steps_dropped_non'])}, total dropped={np.sum(hf_ds['num_steps_dropped_non'])}")
    # 2) build traj -> steps map
    traj2steps = build_traj_step_map_from_qv_jsonl(
        args.qv_jsonl,
        # min_weight=args.min_weight,
        adv_weight_mode=args.adv_weight_mode,
        # adv_beta=args.adv_beta,
        topk_per_traj=args.topk_per_traj,
        drop_zero_weight=args.drop_zero_weight,
        outcome_advantage=args.qa_outcome_advantage and args.task == "qa",
        precomputed_advantage=args.precomputed_advantage,
    )
    print(f"[Index] trajectories with steps: {len(traj2steps)}")
    if len(traj2steps) == 0:
        raise RuntimeError("No samples after filtering. Try lower min_weight / disable topk / disable drop_zero_weight.")

    # weight stats（flatten）
    all_w = [w for  _dict in traj2steps.values() for w in _dict.values()]
    ws = torch.tensor(all_w, dtype=torch.float32)
    print(f"[Weight] mean={ws.mean().item():.6f}, min={ws.min().item():.6f}, p50={ws.median().item():.6f}, p70={ws.kthvalue(int(len(ws)*0.7)).values.item():.6f}, p90={ws.kthvalue(int(len(ws)*0.9)).values.item():.6f}, max={ws.max().item():.6f}")

    global WEIGHT_MEAN, WEIGHT_STD, WEIGHT_MAX, WEIGHT_MIN
    WEIGHT_MEAN = ws.mean().item()
    WEIGHT_STD  = ws.std().item()
    WEIGHT_MAX  = ws.max().item()
    WEIGHT_MIN  = ws.min().item()

    import importlib.util as _ilu
    _attn = "flash_attention_2" if _ilu.find_spec("flash_attn") is not None else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=_attn,
        # device_map="auto",   # 单卡也OK；多卡会自动分配
    )

    # 关键省显存：关 cache
    model.config.use_cache = False

    # if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
        # print("[Model] gradient checkpointing enabled.")

    if args.lora:
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        if args.grad_ckpt and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        model.print_trainable_parameters()

    # 4) dataset + collator（trajectory -> sample K steps -> flatten）
    train_ds = TrajectoryMultiStepDataset(
        hf_ds,
        traj2steps,
        max_length=args.max_length,
    )

    collator = TrajAllStepMaskCollator(
        tokenizer=tokenizer,
        hf_ds=hf_ds,
        traj2steps=traj2steps,
        max_length=args.max_length,
        debug_batch=args.debug_batch,
    )

    # 5) training args
    _world = int(os.environ.get("WORLD_SIZE", "1"))  # FSDP only helps multi-GPU; off for single GPU
    # PEFT's FSDP auto-wrap policy must receive the actual decoder-layer class
    # used by the loaded model (Qwen3DecoderLayer here; LlamaDecoderLayer is
    # not present in Qwen3).
    _no_split_modules = getattr(model, "_no_split_modules", None) or ["Qwen3DecoderLayer"]
    _fsdp_layer_cls = _no_split_modules[0]
    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,   # 这里指“trajectory batch size”
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # save_total_limit=10,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=["wandb"] if args.wandb else [],
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        # gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        fsdp=("full_shard auto_wrap" if _world > 1 else ""),
        fsdp_transformer_layer_cls_to_wrap=_fsdp_layer_cls,
        fsdp_config={
            "use_orig_params": True,
            "limit_all_gathers": True,
            "forward_prefetch": False,
            "activation_checkpointing": True
        },
        warmup_ratio=0.03,
    )

    trainer = WeightedSFTTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        max_tokens_per_device=args.max_tokens_per_device,
        max_examples_per_device=args.max_examples_per_device,
        sort_by_length=(not args.no_length_sort),
        bucket_size=args.bucket_size,
        adv_beta=args.adv_beta,
        neg_mode=args.neg_mode,
        neg_tau=args.neg_tau,
        alpha_ul=args.alpha_ul,
        task=args.task,
        kl_coef=args.kl_coef,
        think_token_weight=args.think_token_weight,  # downweight think tokens
        eps_low=args.eps_low,
        eps_high=args.eps_high,
        ref_sync_steps=args.ref_sync_steps,
        no_reference_model=args.no_reference_model,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
