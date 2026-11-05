# train_traj_step_iql.py
import os
import json
import math
import glob
import time
import random
import argparse
import warnings

import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from packaging import version

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, Sampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModel, Trainer
from transformers.optimization import Adafactor, get_scheduler

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy, logging, is_sagemaker_mp_enabled, is_accelerate_available, is_torch_xla_available, is_apex_available, is_sagemaker_mp_enabled, check_torch_load_is_safe

from trl import SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_from_disk

from eval_agent.data_utils import ChatTemplate, get_chat_template
from aux_reward import assign_aux_rewards
from qa_data import preprocess_traj_dataset_qa
from tqdm import tqdm

logger = logging.get_logger(__name__)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer_callback import  TrainerState, ExportableState
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model

from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_pt_utils import get_model_param_count, reissue_pt_warnings
from transformers.trainer_utils import speed_metrics, TrainOutput
from transformers.trainer import _is_peft_model
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.runtime as xr
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
else:
    IS_XLA_FSDPV2_POST_2_2 = False
    
    
if is_apex_available():
    from apex import amp
    


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_V_NAME = "optimizer_v.pt"
OPTIMIZER_Q_NAME = "optimizer_q.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_V_NAME_BIN = "optimizer_v.bin"
OPTIMIZER_Q_NAME_BIN = "optimizer_q.bin"
SCHEDULER_V_NAME = "scheduler_v.pt"
SCHEDULER_Q_NAME = "scheduler_q.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"



import functools
import contextlib
import shutil

import gc
import tempfile

# -------------------------
# Utils
# -------------------------
def _to_list(x):
    return x.tolist() if isinstance(x, torch.Tensor) else x


def gather_positions(h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    h:  [B, T, H]
    pos:[B, K]  (may contain -1)
    return gathered: [B, K, H]
    """
    B, T, H = h.shape
    pos_safe = pos.clamp_min(0)
    idx = pos_safe.unsqueeze(-1).expand(-1, -1, H)
    return h.gather(dim=1, index=idx)


def print_trainable_parameters(model, name="model"):
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"[Trainable Layer] {n}")
    print(
        f"[Summary] Trainable params: {trainable:,d} | "
        f"Total params: {total:,d} | "
        f"Trainable%: {100 * trainable / total:.4f}%"
    )


# -------------------------
# Preprocess: tokenize full traj once + store end indices per step
# -------------------------
def preprocess_traj_dataset_alfworld(
    data_path: str,
    failed_data_path: Optional[str],
    model_name: str,
    out_dir: Optional[str] = None,
    gamma: float = 0.95,
    max_steps: int = 30,
    max_length: int = 4096,
    use_aux_reward: bool = True,
):

    # data = json.load(open(data_path, "r", encoding="utf-8"))
    # failed = json.load(open(failed_data_path, "r", encoding="utf-8")) if failed_data_path else []
    # data.extend(failed)
    # jsonl
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    chat: ChatTemplate = get_chat_template(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length

    def get_prompt(source_conv):
        roles = {"human": "user", "gpt": "assistant"}
        roles_list = ["user", "assistant"]
        messages = []
        for j, sent in enumerate(source_conv):
            role = roles[sent["from"]]
            assert role == roles_list[j % 2], f"role mismatch at {j}"
            messages.append({"role": role, "content": sent["value"]})
        return chat.get_prompt(messages)

    traj_records = []
    for item in tqdm(data):
        conv = item["conversations"]
        # skip broken/unfinished trajectories (failed env replay -> no terminal, reward-less noise)
        if item.get("meta", {}).get("finished") is False:
            continue
        assigned_rewards = item.get("assigned_reward", None)
        if not use_aux_reward:
            assigned_rewards = None
        elif assigned_rewards is None:
            assigned_rewards = assign_aux_rewards(conv, "alfworld")
        # if sum(assigned_rewards) < 0.0:
        #     print('check')
        # build step lists (same definition as your parse_alfworld_conversation)
        step_states, step_actions, step_nexts = [], [], []
        step_reward, step_done = [], []
        break_flag = 0

        for idx in range(len(conv)):
            if idx != 0 and idx % 2 == 0 and idx + 1 < len(conv):
                state = conv[: idx + 1]
                action = conv[: idx + 2]
                next_state = conv[: idx + 3] if (idx + 2 < len(conv)) else action

                r = assigned_rewards[idx // 2 - 1] if assigned_rewards is not None else 0.0
                step_states.append(state)
                step_actions.append(action)
                step_nexts.append(next_state)
                step_reward.append(float(r))
                step_done.append(0.0)

                # if len(step_states) >= max_steps:
                #     break_flag = 1
                #     break

        T = len(step_states)
        assert T == len(step_actions) == len(step_nexts) == len(step_reward) == len(step_done)
        assert break_flag == 1 or assigned_rewards is None or T == len(assigned_rewards)
        if T == 0:
            continue
        

        
        final_reward = item['meta']['reward'] if item['meta']['reward'] else 0.0
        final_done = step_done[-1] = 1.0 if item['meta']['success'] else 0.0
        won = 1.0 if final_reward >= 1.0 else 0.0   # 真·赢以 reward==1 为准；env-done 死局(success=True 但 reward=0)按失败计权
        step_reward[-1] += final_reward
        weights = np.arange(1, T + 1, dtype=np.float32)
        weights = (weights / len(weights) + won) * 0.5 + 0.5

        # else:
        #     final_reward = step_reward[-1]
        #     final_done = step_done[-1] 
            
            # step_done[-1] = 0.0
            # step_reward[-1] = 0.0
        # tokenize full trajectory once (full conv prompt)
        full_prompt = get_prompt(conv)
        full_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

        # compute end indices for each step by tokenizing prefixes OFFLINE
        state_end, action_end, next_end = [], [], []
        for t in range(T):
            ps = tokenizer(get_prompt(step_states[t]), add_special_tokens=False).input_ids
            pa = tokenizer(get_prompt(step_actions[t]), add_special_tokens=False).input_ids
            pn = tokenizer(get_prompt(step_nexts[t]), add_special_tokens=False).input_ids
            if len(pn) > max_length:
                final_done = 0.0
                print(f"Exceed max length, truncate traj, {len(pn)} > {max_length}, step {t}")
                break
            state_end.append(len(ps) - 1)
            action_end.append(len(pa) - 1)
            next_end.append(len(pn) - 1)

        # trunct
        ep_len = len(state_end)
        last_end = next_end[ep_len - 1]
        traj_records.append(
            {
                "input_ids": full_ids[: last_end + 1],
                "state_end": state_end,
                "action_end": action_end,
                "next_end": next_end,
                "reward": step_reward[:ep_len],
                "done": step_done[:ep_len],
                "weight": weights[:ep_len],
                # "q_label": q_label,
                "final_done": int(final_done),
                # "is_fail": int(is_fail),
                "ep_len": int(ep_len),
            }
        )

    ds = Dataset.from_list(traj_records)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[Preprocess] Saved {len(ds)} trajectories to {out_dir}")
    return ds

def preprocess_traj_dataset_sciworld(
    data_path: str,
    failed_data_path: Optional[str],
    model_name: str,
    out_dir: Optional[str] = None,
    gamma: float = 0.95,
    max_steps: int = 24,
    max_length: int = 4096,
    use_aux_reward: bool = True,
):

    # data = json.load(open(data_path, "r", encoding="utf-8"))
    # failed = json.load(open(failed_data_path, "r", encoding="utf-8")) if failed_data_path else []
    # data.extend(failed)
    # jsonl
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    chat: ChatTemplate = get_chat_template(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length

    def get_prompt(source_conv):
        roles = {"human": "user", "gpt": "assistant"}
        roles_list = ["user", "assistant"]
        messages = []
        for j, sent in enumerate(source_conv):
            role = roles[sent["from"]]
            assert role == roles_list[j % 2], f"role mismatch at {j}"
            messages.append({"role": role, "content": sent["value"]})
        return chat.get_prompt(messages)

    traj_records = []
    for item in tqdm(data):
        conv = item["conversations"]
        # skip broken/unfinished trajectories (failed env replay -> no terminal, reward-less noise)
        if item.get("meta", {}).get("finished") is False:
            continue
        assigned_rewards = item.get("assigned_reward", None)
        if not use_aux_reward:
            assigned_rewards = None
        elif assigned_rewards is None:
            assigned_rewards = assign_aux_rewards(conv, "sciworld")
        # if sum(assigned_rewards) < 0.0:
        #     print('check')
        # build step lists (same definition as your parse_alfworld_conversation)
        step_states, step_actions, step_nexts = [], [], []
        step_reward, step_done = [], []
        break_flag = 0

        for idx in range(len(conv)):
            if idx != 0 and idx % 2 == 0 and idx + 1 < len(conv):
                state = conv[: idx + 1]
                action = conv[: idx + 2]
                next_state = conv[: idx + 3] if (idx + 2 < len(conv)) else action

                r = assigned_rewards[idx // 2 - 1] if assigned_rewards is not None else 0.0
                step_states.append(state)
                step_actions.append(action)
                step_nexts.append(next_state)
                step_reward.append(float(r))
                step_done.append(0.0)

                # if len(step_states) >= max_steps:
                #     break_flag = 1
                #     break

        T = len(step_states)
        assert T == len(step_actions) == len(step_nexts) == len(step_reward) == len(step_done)
        assert break_flag == 1 or assigned_rewards is None or T == len(assigned_rewards)
        if T == 0:
            continue
        

        
        final_reward = item['meta']['reward'] if item['meta']['reward'] else 0.0
        final_done = step_done[-1] = 1.0 if item['meta']['terminate_reason'] == 'success' else 0.0
        won = 1.0 if final_reward >= 1.0 else 0.0   # weight by a TRUE win (reward==1.0), not env terminate_reason (matches alfworld)
        step_reward[-1] += final_reward
        weights = np.arange(1, T + 1, dtype=np.float32)
        weights = (weights / len(weights) + won) * 0.5 + 0.5

        # else:
        #     final_reward = step_reward[-1]
        #     final_done = step_done[-1] 
            
            # step_done[-1] = 0.0
            # step_reward[-1] = 0.0
        # tokenize full trajectory once (full conv prompt)
        full_prompt = get_prompt(conv)
        full_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

        # compute end indices for each step by tokenizing prefixes OFFLINE
        state_end, action_end, next_end = [], [], []
        for t in range(T):
            ps = tokenizer(get_prompt(step_states[t]), add_special_tokens=False).input_ids
            pa = tokenizer(get_prompt(step_actions[t]), add_special_tokens=False).input_ids
            pn = tokenizer(get_prompt(step_nexts[t]), add_special_tokens=False).input_ids
            if len(pn) > max_length:
                final_done = 0.0
                print(f"Exceed max length, truncate traj, {len(pn)} > {max_length}, step {t}")
                break
            state_end.append(len(ps) - 1)
            action_end.append(len(pa) - 1)
            next_end.append(len(pn) - 1)

        # trunct
        ep_len = len(state_end)
        last_end = next_end[ep_len - 1]
        traj_records.append(
            {
                "input_ids": full_ids[: last_end + 1],
                "state_end": state_end,
                "action_end": action_end,
                "next_end": next_end,
                "reward": step_reward[:ep_len],
                "done": step_done[:ep_len],
                "weight": weights[:ep_len],
                # "q_label": q_label,
                "final_done": int(final_done),
                # "is_fail": int(is_fail),
                "ep_len": int(ep_len),
            }
        )

    ds = Dataset.from_list(traj_records)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[Preprocess] Saved {len(ds)} trajectories to {out_dir}")
    return ds


def preprocess_traj_dataset_webshop(
    data_path: str,
    failed_data_path: Optional[str],
    model_name: str,
    out_dir: Optional[str] = None,
    gamma: float = 0.95,
    max_steps: int = 6,
    max_length: int = 4096,
    use_aux_reward: bool = True,
):

    # data = json.load(open(data_path, "r", encoding="utf-8"))
    # failed = json.load(open(failed_data_path, "r", encoding="utf-8")) if failed_data_path else []
    # data.extend(failed)
    # jsonl
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    chat: ChatTemplate = get_chat_template(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length

    def get_prompt(source_conv):
        roles = {"human": "user", "gpt": "assistant"}
        roles_list = ["user", "assistant"]
        messages = []
        for j, sent in enumerate(source_conv):
            role = roles[sent["from"]]
            assert role == roles_list[j % 2], f"role mismatch at {j}"
            messages.append({"role": role, "content": sent["value"]})
        return chat.get_prompt(messages)

    traj_records = []
    for item in tqdm(data):
        conv = item["conversations"]
        # skip broken/unfinished trajectories (failed env replay -> no terminal, reward-less noise)
        if item.get("meta", {}).get("finished") is False:
            continue
        assigned_rewards = item.get("assigned_reward", None)
        if not use_aux_reward:
            assigned_rewards = None
        elif assigned_rewards is None:
            assigned_rewards = assign_aux_rewards(conv, "webshop")
        # if sum(assigned_rewards) < 0.0:
        #     print('check')
        # build step lists (same definition as your parse_alfworld_conversation)
        step_states, step_actions, step_nexts = [], [], []
        step_reward, step_done = [], []
        break_flag = 0

        for idx in range(len(conv)):
            if idx != 0 and idx % 2 == 0 and idx + 1 < len(conv):
                state = conv[: idx + 1]
                action = conv[: idx + 2]
                next_state = conv[: idx + 3] if (idx + 2 < len(conv)) else action

                r = assigned_rewards[idx // 2 - 1] if assigned_rewards is not None else 0.0
                step_states.append(state)
                step_actions.append(action)
                step_nexts.append(next_state)
                step_reward.append(float(r))
                step_done.append(0.0)

                # if len(step_states) >= max_steps:
                #     break_flag = 1
                #     break

        T = len(step_states)
        assert T == len(step_actions) == len(step_nexts) == len(step_reward) == len(step_done)
        assert break_flag == 1 or assigned_rewards is None or T == len(assigned_rewards)
        if T == 0:
            continue
        

        
        final_reward = item['meta']['reward'] if item['meta']['reward'] else 0.0
        final_done = step_done[-1] = 1.0 if item['meta']['terminate_reason'] == 'success' else 0.0
        step_reward[-1] += final_reward

        # else:
        #     final_reward = step_reward[-1]
        #     final_done = step_done[-1] 
            
            # step_done[-1] = 0.0
            # step_reward[-1] = 0.0
        # tokenize full trajectory once (full conv prompt)
        full_prompt = get_prompt(conv)
        full_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids

        # compute end indices for each step by tokenizing prefixes OFFLINE
        state_end, action_end, next_end = [], [], []
        for t in range(T):
            ps = tokenizer(get_prompt(step_states[t]), add_special_tokens=False).input_ids
            pa = tokenizer(get_prompt(step_actions[t]), add_special_tokens=False).input_ids
            pn = tokenizer(get_prompt(step_nexts[t]), add_special_tokens=False).input_ids
            if len(pn) > max_length:
                final_done = 0.0
                print(f"Exceed max length, truncate traj, {len(pn)} > {max_length}, step {T} - {t}")
                break
            state_end.append(len(ps) - 1)
            action_end.append(len(pa) - 1)
            next_end.append(len(pn) - 1)

        T = len(state_end)
        won = 1.0 if final_reward >= 1.0 else 0.0   # weight by a TRUE win (reward==1), matches sciworld/alfworld
        weights = np.arange(1, T + 1, dtype=np.float32)
        weights = (weights / len(weights) + won) * 0.5 + 0.5

        # trunct
        ep_len = len(state_end)
        last_end = next_end[ep_len - 1]
        traj_records.append(
            {
                "input_ids": full_ids[: last_end + 1],
                "state_end": state_end,
                "action_end": action_end,
                "next_end": next_end,
                "reward": step_reward[:ep_len],
                "done": step_done[:ep_len],
                "weight": weights[:ep_len],
                # "q_label": q_label,
                "final_done": int(final_done),
                # "is_fail": int(is_fail),
                "ep_len": int(ep_len),
            }
        )

    ds = Dataset.from_list(traj_records)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[Preprocess] Saved {len(ds)} trajectories to {out_dir}")
    return ds


# -------------------------
# Dispatcher: build the in-memory trajectory dataset for a task.
# Pass out_dir to also cache it to disk (optional; off by default).
# -------------------------
def build_traj_dataset(
    task: str,
    data_path: str,
    model_name: str,
    failed_data_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    gamma: float = 0.95,
    max_length: int = 4096,
    use_aux_reward: bool = True,
):
    if task == "alfworld":
        fn = preprocess_traj_dataset_alfworld
    elif task == "sciworld":
        fn = preprocess_traj_dataset_sciworld
    elif task == "webshop":
        fn = preprocess_traj_dataset_webshop
    elif task == "qa":
        fn = preprocess_traj_dataset_qa
    else:
        raise NotImplementedError(f"unknown task: {task}")
    return fn(
        data_path=data_path,
        failed_data_path=failed_data_path,
        model_name=model_name,
        out_dir=out_dir,
        gamma=gamma,
        max_length=max_length,
        use_aux_reward=use_aux_reward,
    )


# -------------------------
# Trajectory sampling dataset: sample K=4 steps per trajectory + return full prefix
# -------------------------
class TrajStepDataset(TorchDataset):
    def __init__(
        self,
        hf_ds,
        steps_per_traj: int = 4,
        max_length: int = 4096,
        seed: int = 0,
    ):
        self.ds = hf_ds
        self.K = steps_per_traj
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        T = int(item["ep_len"])

        rng = random.Random(self.seed + self.epoch * 1000003 + i * 9176)

        # candidate steps whose prefix doesn't exceed max_length
        cand = [t for t in range(T) if item["next_end"][t] < self.max_length]
        if len(cand) == 0:
            cand = [0]

        if len(cand) >= self.K:
            steps = rng.sample(cand, self.K)
        else:
            steps = [rng.choice(cand) for _ in range(self.K)]
        steps = sorted(steps)

        # "全部 prefix" for these steps: cut to the longest next_end among sampled steps
        cut_end = max(item["next_end"][t] for t in steps)
        cut_end = min(cut_end, self.max_length - 1)

        input_ids = item["input_ids"][: cut_end + 1]

        def pick(arr):
            return [arr[t] for t in steps]

        return {
            "input_ids": input_ids,
            "state_pos": pick(item["state_end"]),
            "action_pos": pick(item["action_end"]),
            "next_pos": pick(item["next_end"]),
            "reward": pick(item["reward"]),
            "done": pick(item["done"]),
            "weight": pick(item["weight"]),
            # "q_label": pick(item["q_label"]),
            "step_id": steps,
            "ep_len": T,
            "final_done": int(item["final_done"]),
            "K": self.K,
        }



class TrajChunkDataset(TorchDataset):
    """
    每个样本 = (traj_id, K个step_id)；K=4
    每个 epoch 重建 index，保证覆盖；chunk 内 step 分布在不同时间段（不连续）
    """
    def __init__(self, hf_ds, K=4, max_length=4096, seed=0, pad_last=True):
        self.ds = hf_ds
        self.K = K
        self.max_length = max_length
        self.seed = seed
        self.pad_last = pad_last
        self.epoch = 0
        self.index = []
        self._build_index()

    def set_epoch(self, epoch:int):
        self.epoch = int(epoch)
        self._build_index()

    def _build_index(self):
        self.index = []
        self.sample_lens = []   # <<< 新增：与 self.index 对齐
        self.has_last = [] 
        rng = random.Random(self.seed + self.epoch * 1000003)
        K = self.K

        for traj_id in range(len(self.ds)):
            item = self.ds[traj_id]
            T = int(item["ep_len"])
            if T <= 0:
                continue

            segments = []
            for k in range(K):
                lo = (T * k) // K
                hi = (T * (k + 1)) // K
                seg = list(range(lo, hi))
                rng.shuffle(seg)
                segments.append(seg)

            max_len = max(len(seg) for seg in segments)
            last = T - 1

            for j in range(max_len):
                chunk = []
                for k in range(K):
                    if j < len(segments[k]):
                        chunk.append(segments[k][j])
                    else:
                        chunk.append(last if self.pad_last else segments[k][-1])

                cut_end = max(item["next_end"][t] for t in chunk)
                if cut_end >= self.max_length:
                    continue

                self.index.append((traj_id, chunk))
                self.sample_lens.append(int(cut_end) + 1)   # <<< 新增：token length
                
                self.has_last.append(1 if (last in chunk) else 0)   # <<< 新增

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_id, steps = self.index[idx]
        item = self.ds[traj_id]

        cut_end = max(item["next_end"][t] for t in steps)
        cut_end = min(cut_end, self.max_length - 1)

        input_ids = item["input_ids"][:cut_end+1]

        def pick(arr):
            return [arr[t] for t in steps]

        return {
            "input_ids": input_ids,
            "state_pos": pick(item["state_end"]),
            "action_pos": pick(item["action_end"]),
            "next_pos": pick(item["next_end"]),
            "reward": pick(item["reward"]),
            "done": pick(item["done"]),
            "weight": pick(item["weight"]),
            # "q_label": pick(item["q_label"]),
            "step_id": steps,
            "ep_len": int(item["ep_len"]),
            "final_done": int(item["final_done"]),
            "K": self.K,
        }



@dataclass
class TrajStepCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        K = int(features[0]["K"])

        batch_tok = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [{
                "input_ids": _to_list(f["input_ids"]),
                "attention_mask": [1] * len(f["input_ids"]),
            } for f in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        def stack_int(name, fill=-1):
            x = torch.full((len(features), K), fill, dtype=torch.long)
            for i, f in enumerate(features):
                x[i] = torch.tensor(f[name], dtype=torch.long)
            return x

        def stack_float(name, fill=0.0):
            x = torch.full((len(features), K), fill, dtype=torch.float32)
            for i, f in enumerate(features):
                x[i] = torch.tensor(f[name], dtype=torch.float32)
            return x

        state_pos  = stack_int("state_pos")
        action_pos = stack_int("action_pos")
        next_pos   = stack_int("next_pos")
        step_id    = stack_int("step_id")

        reward = stack_float("reward").unsqueeze(-1)   # [B,K,1]
        done   = stack_float("done").unsqueeze(-1)
        weight = stack_float("weight", 1.0)            # [B,K]
        # q_label= stack_float("q_label").unsqueeze(-1)

        ep_len = torch.tensor([f["ep_len"] for f in features], dtype=torch.long)
        final_done = torch.tensor([f["final_done"] for f in features], dtype=torch.float32).view(-1,1)

        step_mask = (next_pos >= 0).float()

        return {
            "input_ids": batch_tok["input_ids"],
            "attention_mask": batch_tok["attention_mask"],
            "state_pos": state_pos,
            "action_pos": action_pos,
            "next_pos": next_pos,
            "step_id": step_id,
            "reward": reward,
            "done": done,
            "weight": weight,
            # "q_label": q_label,
            "ep_len": ep_len,
            "final_done": final_done,
            "step_mask": step_mask,
        }


class TokenBudgetBatchSampler(Sampler[List[int]]):
    """
    动态 batch：每个 batch 的总 token 数 <= max_tokens
    兼容 DDP：先在“全局样本序列”上构造 batches，再 pad 到 world_size 的倍数，
             然后每个 rank 取自己那份 batch（batch 数完全一致，避免 hang）。
    """
    def __init__(
        self,
        lengths: List[int],
        max_tokens: int,
        max_examples: int = 999999,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        warmup_epochs: int = 0,
        warmup_mask: bool = False,
        # bucket_size: int = 256,
    ):
        self.lengths = lengths
        self.max_tokens = int(max_tokens)
        self.max_examples = int(max_examples)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.drop_last = drop_last

        # self.bucket_size = int(bucket_size)

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.warmup_epochs = int(warmup_epochs)
        self.warmup_mask = warmup_mask  
        self.epoch = 0
        self._batches_rank = []

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._build_batches()

    def _build_batches(self):
        n = len(self.lengths)


        if self.warmup_mask is not None and self.epoch < self.warmup_epochs:
            indices = [i for i in range(n) if int(self.warmup_mask[i]) == 1]
            if len(indices) == 0:
                indices = list(range(n))  # 兜底
        else:
            indices = list(range(n))

        rng = random.Random(self.seed + self.epoch * 1000003)

        if self.shuffle:
            rng.shuffle(indices)  # ✅ 只有 shuffle，没有长度排序

        global_batches = []
        cur = []
        cur_max = 0  # 当前 batch 的 max seq len（决定 padding 后成本）

        for idx in indices:
            L = int(self.lengths[idx])
            if L <= 0:
                continue

            # 单条太长：单独成 batch（padding-aware 情况也必须这么做）
            if L > self.max_tokens:
                if cur:
                    global_batches.append(cur)
                global_batches.append([idx])
                cur = []
                cur_max = 0
                continue

            new_max = max(cur_max, L)
            projected = (len(cur) + 1) * new_max  # ✅ 关键：padding-aware 预算
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
# Model: keep your v/q/delayed adapters, but allow multi-step positions
# -------------------------
class ProbeModel(nn.Module):
    """
    AutoModel backbone + heads.
    Keep v_adapter / q_adapter / delayed_q_adapter.
    Multi-step training: gather positions [B,K] in one forward.
    """
    def __init__(
        self,
        model_name: str,
        lora: LoraConfig = None,
        max_length=4096,
        use_done_head=True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = max_length

        import importlib.util as _ilu
        _attn = "flash_attention_2" if _ilu.find_spec("flash_attn") is not None else "sdpa"
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=_attn,
        )

        # 1) 训练必须关 KV cache
        self.backbone.config.use_cache = False

        # 2) 省激活显存（长 seq 强烈建议）。可用 GC=0 关闭以提速（flash-attn 下显存够时）。
        _use_gc = os.environ.get("GC", "1") == "1"
        if _use_gc and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            print(f"[ProbeModel] Enabled gradient checkpointing for backbone.")
        base = getattr(self.backbone, "base_model", None)
        if base is not None and hasattr(base, "config"):
            base.config.use_cache = False
            print(f"[ProbeModel] Disabled use_cache for base model.")
            if _use_gc and hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
                print(f"[ProbeModel] Enabled gradient checkpointing for base model.")
        if not _use_gc:
            print(f"[ProbeModel] Gradient checkpointing DISABLED (GC=0).")
                

        self.v_adapter = "v_adapter"
        self.q_adapter = "q_adapter"
        self.delayed_q_adapter = "delayed_q_adapter"
        self.use_done_head = use_done_head

        if lora is not None:
            self.backbone = get_peft_model(self.backbone, lora, adapter_name=self.v_adapter)
            self.backbone.add_adapter(self.q_adapter, lora)
            self.backbone.add_adapter(self.delayed_q_adapter, lora)

            if hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "prepare_inputs_for_generation"):
                self.backbone.base_model.prepare_inputs_for_generation = None

        hidden_size = self.backbone.config.hidden_size

        def mlp():
            return nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1, bias=False),
            )

        self.head_v = mlp()
        self.head_q1 = mlp()
        self.head_q2 = mlp()
        if self.use_done_head:
            self.done_head = mlp()  # logits

        self.delayed_head_q1 = mlp()
        self.delayed_head_q2 = mlp()

        self.sync_delayed_q()
        self.freeze_delayed_q() 
        self.unfreeze_adapter(self.v_adapter)
        self.unfreeze_adapter(self.q_adapter)

    def unfreeze_adapter(self, adapter_name: str):
        for n, p in self.named_parameters():
            if f".{adapter_name}." in n:
                p.requires_grad_(True)

    def _forward_backbone(self, adapter_name, input_ids, attention_mask):
        old = getattr(self.backbone, "active_adapter", None)
        self.backbone.set_adapter(adapter_name)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if old is not None:
            self.backbone.set_adapter(old)
        return out.last_hidden_state  # [B,T,H]

    def forward_v_multi(self, input_ids, attention_mask, state_pos, next_pos):
        h = self._forward_backbone(self.v_adapter, input_ids, attention_mask)
        hs = gather_positions(h, state_pos)  # [B,K,H]
        hn = gather_positions(h, next_pos)   # [B,K,H]
        V = self.head_v(hs.float())                  # [B,K,1]
        Vn = self.head_v(hn.float())                 # [B,K,1]
        return {"V": V, "V_next": Vn}

    def forward_q_multi(self, input_ids, attention_mask, action_pos):
        h = self._forward_backbone(self.q_adapter, input_ids, attention_mask)
        ha = gather_positions(h, action_pos)  # [B,K,H]
        Q1 = self.head_q1(ha.float())
        Q2 = self.head_q2(ha.float())
        out = {"Q1": Q1, "Q2": Q2}
        if self.use_done_head:
            out["done_logits"] = self.done_head(ha)  # [B,K,1]
        return out

    @torch.no_grad()
    def infer_delayed_q_multi(self, input_ids, attention_mask, action_pos):
        h = self._forward_backbone(self.delayed_q_adapter, input_ids, attention_mask)
        ha = gather_positions(h, action_pos)
        Q1 = self.delayed_head_q1(ha.float())
        Q2 = self.delayed_head_q2(ha.float())
        return {"dQ1": Q1, "dQ2": Q2}

    @torch.no_grad()
    def sync_delayed_q(self):
        self.delayed_head_q1.load_state_dict(self.head_q1.state_dict())
        self.delayed_head_q2.load_state_dict(self.head_q2.state_dict())
        self._update_delayed_adapter(tau=1.0)

    @torch.no_grad()
    def soft_update_delayed_q(self, tau: float = 0.05):
        for p_tgt, p_src in zip(self.delayed_head_q1.parameters(), self.head_q1.parameters()):
            p_tgt.data.mul_(1 - tau).add_(p_src.data, alpha=tau)
        for p_tgt, p_src in zip(self.delayed_head_q2.parameters(), self.head_q2.parameters()):
            p_tgt.data.mul_(1 - tau).add_(p_src.data, alpha=tau)
        self._update_delayed_adapter(tau=tau)

    @torch.no_grad()
    def _update_delayed_adapter(self, tau: float):
        params = dict(self.named_parameters())
        q_tag = f".{self.q_adapter}."
        dq_tag = f".{self.delayed_q_adapter}."
        for name, p in params.items():
            if q_tag in name:
                dst_name = name.replace(q_tag, dq_tag)
                if dst_name in params:
                    dst = params[dst_name]
                    dst.data.mul_(1 - tau).add_(p.data, alpha=tau)

    @torch.no_grad()
    def freeze_delayed_q(self):
        for n, p in self.named_parameters():
            if n.startswith("delayed_head_q1.") or n.startswith("delayed_head_q2."):
                p.requires_grad_(False)
            if f".{self.delayed_q_adapter}." in n:
                p.requires_grad_(False)



# -------------------------
# Trainer: multi-step loss with K=4
# -------------------------
class QTrainer(Trainer):
    def __init__(self, gamma=0.95, tau=0.7, done_last_k=3, use_done_head=True, max_tokens_per_device=None, warmup_epochs=1, **kwargs):
        super().__init__(**kwargs)
        self._gamma = gamma
        self._tau = tau
        self.done_last_k = done_last_k
        self.use_done_head = use_done_head
        self.max_tokens_per_device = max_tokens_per_device
        self.warmup_epochs = warmup_epochs
        
        
        self._stage = "boost_q"
        self.optimizer_q = None
        self.optimizer_v = None
        self.lr_scheduler_q = None
        self.lr_scheduler_v = None

    @staticmethod
    def expectile_loss(diff: torch.Tensor, tau: float, mask: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """
        diff: [B,K,1]
        mask: [B,K] (0/1)
        weight: [B,K] optional
        """
        w = torch.where(diff < 0, 1.0 - tau, tau)  # broadcast
        per = (w * diff.pow(2)).squeeze(-1)  # [B,K]
        # per = per * weight if weight is not None else per
        loss = (per * mask).sum() / mask.sum().clamp_min(1.0)
        if weight is not None:
            weighted_loss = (per * mask * weight).sum() / mask.sum().clamp_min(1.0)
            return weighted_loss, loss
        return None, loss
            

    def _allreduce_grads(self, model):
        """FIX(DDP): compute_loss forwards through `model.module` (custom forward_q/v_multi
        methods that DDP's wrapper cannot intercept), so DDP's autograd hooks never fire and
        NO gradient all-reduce happens -> the 4 ranks trained independent replicas and only
        rank 0's was saved. Manually average grads across ranks before the optimizer step so
        every rank applies the same, data-parallel-correct gradient (and grad-clip sees the
        global norm). Only the active stage's params carry grads; None grads are skipped."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        ws = dist.get_world_size()
        if ws <= 1:
            return
        _dbg = os.environ.get("ALLREDUCE_DEBUG") == "1" and self.state.global_step < 3
        def _global_gnorm():
            s = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    s += float(p.grad.detach().float().pow(2).sum())
            return s ** 0.5
        if _dbg:
            pre = _global_gnorm()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ws)
        if _dbg:
            post = _global_gnorm()
            # Expect: pre DIFFERS across ranks (each saw its own batch shard); post IDENTICAL.
            print(f"[ALLREDUCE_DEBUG] rank={dist.get_rank()} step={self.state.global_step} "
                  f"global_grad_norm pre={pre:.5f} post={post:.5f}", flush=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # 交替 stage（跟你原来一样）
        if self.state.global_step % 2 == 0:
            self._stage = stage = "boost_q"
        else:
            self._stage = stage = "expectile_v"

        step_mask = inputs["step_mask"]  # [B,K]
        
        # ================= boost_q：TD 学 Q =================
        if stage == "boost_q":
            inner = model.module if hasattr(model, "module") else model
            inner.train()
            out_q = inner.forward_q_multi(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                action_pos=inputs["action_pos"],
            )
            Q1, Q2 = out_q["Q1"].float(), out_q["Q2"].float()  # [B,K,1]

            # done loss（可选）：仍按你原逻辑，只在 episode 最后 K 步监督
            done_loss = None
            if self.use_done_head and ("done_logits" in out_q):
                done_logits = out_q["done_logits"].squeeze(-1)  # [B,K]
                ep_len = inputs["ep_len"].view(-1, 1).float()   # [B,1]
                # step_id: [B,K]，只监督最后 done_last_k 步（你原来 K 未定义，这里用 self.done_last_k）
                last_mask = (inputs["step_id"].float() >= (ep_len - float(self.done_last_k))).float()  # [B,K]
                m = step_mask * last_mask

                tgt = inputs["final_done"].expand_as(done_logits).to(done_logits.dtype)
                per_bce = torch.nn.functional.binary_cross_entropy_with_logits(done_logits, tgt, reduction="none")
                denom2 = m.sum().clamp_min(1.0)
                done_loss = (per_bce * m).sum() / denom2

            # V_next target
            with torch.inference_mode():
                inner.eval()
                out_v = inner.forward_v_multi(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    state_pos=inputs["state_pos"],
                    next_pos=inputs["next_pos"],
                )
            V_next = out_v["V_next"].detach().float()  # [B,K,1]
            inner.train()

            q_target = inputs["reward"] + (1.0 - inputs["done"]) * self._gamma * V_next  # [B,K,1]

            per1 = (Q1 - q_target).pow(2).squeeze(-1)  
            per2 = (Q2 - q_target).pow(2).squeeze(-1) 
            denom = step_mask.sum().clamp_min(1.0)
            q1_loss = (per1 * step_mask).sum() / denom 
            q2_loss = (per2 * step_mask).sum() / denom 
            weighted_q1_loss = (per1 * inputs["weight"] * step_mask).sum() / denom
            weighted_q2_loss = (per2 * inputs["weight"] * step_mask).sum() / denom
            loss = weighted_q1_loss + weighted_q2_loss

            if done_loss is not None:
                loss = loss + 0.1 * done_loss

            # if random.random() < 0.1:
            if self.state.global_step % 20 != 0:   # skip per-step .item() GPU syncs (logging only)
                self.dict_to_log = {}
                return (loss, out_q) if return_outputs else loss
            log_dict = {
                "batch_size": inputs["input_ids"].size(0),
                "q_loss": loss.detach().float().item(),
                
                "q1_loss": q1_loss.detach().float().item(),
                "q2_loss": q2_loss.detach().float().item(),
                "weighted_q1_loss": weighted_q1_loss.detach().float().item(),
                "weighted_q2_loss": weighted_q2_loss.detach().float().item(),
                "Q1_mean": Q1.detach().float().mean().item(),
                
                "Q2_mean": Q2.detach().float().mean().item(),
                "Q1_std": Q1.detach().float().std(unbiased=False).item(),
                "Q2_std": Q2.detach().float().std(unbiased=False).item(),
                "tgt_nextV_mean": V_next.detach().float().mean().item(),
                "tgt_nextV_std": V_next.detach().float().std(unbiased=False).item(),
                "q_target_mean": q_target.detach().float().mean().item(),
                "q_target_std": q_target.detach().float().std(unbiased=False).item(),
                "step_mask_mean": step_mask.detach().float().mean().item(),
                "weight_mean": inputs["weight"].detach().float().mean().item(),
                "reward_mean": inputs["reward"].detach().float().mean().item(),
            }
            if done_loss is not None:
                log_dict["done_loss"] = done_loss.detach().float().item()
            # self.log(log_dict, self.state.global_step)
            log_dict = {f"{self._stage}/{k}": v for k, v in log_dict.items()}
            self.dict_to_log = log_dict  # 延后到 training_step_end 里打印
            return (loss, out_q) if return_outputs else loss

        # ================= expectile_v：学 V =================
        inner = model.module if hasattr(model, "module") else model
        inner.train()
        out_v = inner.forward_v_multi(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            state_pos=inputs["state_pos"],
            next_pos=inputs["next_pos"],
        )
        V = out_v["V"].float()  # [B,K,1]

        with torch.inference_mode():
            inner.eval()
            d = inner.infer_delayed_q_multi(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                action_pos=inputs["action_pos"],
            )
            Q1_tgt = d["dQ1"]
            Q2_tgt = d["dQ2"]
        Q_tgt = torch.minimum(Q1_tgt, Q2_tgt).detach().float()  # [B,K,1]
        model.train()

        weighted_v_loss, v_loss = self.expectile_loss(Q_tgt - V, tau=self._tau, mask=step_mask, weight=inputs["weight"])
        loss = weighted_v_loss 

        if self.state.global_step % 20 != 0:   # skip per-step .item() GPU syncs (logging only)
            self.dict_to_log = {}
            return (loss, out_v) if return_outputs else loss
        log_dict =  {
                "batch_size": inputs["input_ids"].size(0),
                "v_loss": v_loss.detach().float().item(),
                "weighted_loss":  weighted_v_loss.detach().float().item(),
                "V_mean": V.detach().float().mean().item(),
                "V_std": V.detach().float().std(unbiased=False).item(),
                "Q_tgt_mean": Q_tgt.detach().float().mean().item(),
                "Q_tgt_std": Q_tgt.detach().float().std(unbiased=False).item(),
                "Q1_tgt_mean": Q1_tgt.detach().float().mean().item(),
                "Q2_tgt_mean": Q2_tgt.detach().float().mean().item(),
                "Q1_tgt_std": Q1_tgt.detach().float().std(unbiased=False).item(),
                "Q2_tgt_std": Q2_tgt.detach().float().std(unbiased=False).item(),
                "step_mask_mean": step_mask.detach().float().mean().item(),
                "weight_mean": inputs["weight"].detach().float().mean().item(),
                "reward_mean": inputs["reward"].detach().float().mean().item(),
            }
        log_dict = {f"{self._stage}/{k}": v for k, v in log_dict.items()}
        self.dict_to_log = log_dict  # 延后到 training_step_end 里打印
        return (loss, out_v) if return_outputs else loss


    def get_train_dataloader(self):
        # 每个 epoch：先让 dataset 重建 chunk（覆盖），并更新 sample_lens
        if hasattr(self.train_dataset, "set_epoch") and hasattr(self, "state"):
            self.train_dataset.set_epoch(int(self.state.epoch or 0))

        # ===== 动态 token batch =====
        # 你可以在 QTrainer.__init__ 加一个 self.max_tokens_per_device
        # 或者直接写死一个值（比如 8192 / 12288 / 16384）
        max_tokens = getattr(self, "max_tokens_per_device", 12288)

        lengths = getattr(self.train_dataset, "sample_lens", None)
        has_last = getattr(self.train_dataset, "has_last", None)

        # lengths 来自 TrajChunkDataset.sample_lens
        lengths = getattr(self.train_dataset, "sample_lens", None)
        if lengths is None:
            # 兜底：现场算（不推荐）
            lengths = [len(self.train_dataset[i]["input_ids"]) for i in range(len(self.train_dataset))]

        batch_sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            max_examples=32,  # 也可设置一个上限，比如 8/16
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
            warmup_epochs=self.warmup_epochs,
            warmup_mask=has_last,
            # bucket_size=256,
        )
        batch_sampler.set_epoch(int(self.state.epoch or 0))

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,   # <<< 关键：用 batch_sampler，不用 batch_size/sampler
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}
            
            logs.update(getattr(self, "dict_to_log", {}))  # 来自 compute_loss 的日志
            
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs[f"{self._stage}/training_loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs[f"{self._stage}/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs[f"{self._stage}/learning_rate"] = learning_rate
            else:
                logs[f"{self._stage}/learning_rate"] = self._get_learning_rate()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # EMA update delayed nets periodically
        inner = model.module if hasattr(model, "module") else model
        if self.state.global_step % 2 == 0:
            inner.soft_update_delayed_q(tau=0.005)


    def _split_qv_params(self, model):
        """
        返回两组参数：Q(含 q_adapter + head_q1/q2 + done_head) 和 V(含 v_adapter + head_v)
        """
        inner = model.module if hasattr(model, "module") else model

        # 取 adapter 名（你 ProbeModel 里定义的）
        v_adapter = getattr(inner, "v_adapter", "v_adapter")
        q_adapter = getattr(inner, "q_adapter", "q_adapter")

        q_params, v_params = [], []
        for n, p in inner.named_parameters():
            if not p.requires_grad:
                continue

            is_v = (f".{v_adapter}." in n) or n.startswith("head_v.")
            is_q = (f".{q_adapter}." in n) or n.startswith("head_q1.") or n.startswith("head_q2.") or n.startswith("done_head.")

            # 保险：delayed 永远不进 optimizer
            if "delayed_" in n or f".delayed_q_adapter." in n:
                continue

            if is_v:
                v_params.append(p)
            if is_q:
                q_params.append(p)

        if len(q_params) == 0 or len(v_params) == 0:
            raise RuntimeError(f"Split params failed: q={len(q_params)}, v={len(v_params)}. Check name rules / requires_grad.")

        return q_params, v_params

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        建两个 optimizer + 两个 scheduler。Trainer 仍然会调用这个函数一次。
        """
        model = self.model

        q_params, v_params = self._split_qv_params(model)

        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # 两套 optimizer：超参先保持一致（最小改动）
        self.optimizer_q = opt_cls([{"params": q_params}], **opt_kwargs)
        self.optimizer_v = opt_cls([{"params": v_params}], **opt_kwargs)

        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        self.lr_scheduler_q = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer_q,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.lr_scheduler_v = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer_v,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # 给 Trainer 挂一个“占位”的 optimizer/scheduler，避免内部某些地方取 self.optimizer/self.lr_scheduler 报错
        self.optimizer = self.optimizer_q
        self.lr_scheduler = self.lr_scheduler_q

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """
        Trainer 的 step hook：我们忽略传进来的 optimizer，改为按 stage 选择 Q 或 V 的 optimizer。
        """
        stage = getattr(self, "_stage", "boost_q")
        if stage == "boost_q":
            opt = self.optimizer_q
            sch = self.lr_scheduler_q
        else:
            opt = self.optimizer_v
            sch = self.lr_scheduler_v

        # accelerate 兼容
        if hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.step(opt)
        else:
            opt.step()

        if sch is not None:
            sch.step()

        # 两个都清掉更安全（避免某些参数意外残留 grad）
        opt.zero_grad(set_to_none=True)
        if stage == "boost_q":
            self.optimizer_v.zero_grad(set_to_none=True)
        else:
            self.optimizer_q.zero_grad(set_to_none=True)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer_v, self.optimizer_q = self.accelerator.prepare(self.model, self.optimizer_v, self.optimizer_q)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer_v, self.lr_scheduler, self.optimizer_q = self.accelerator.prepare(
                    self.model, self.optimizer_v, self.lr_scheduler, self.optimizer_q
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer_v = self.accelerator.prepare(self.optimizer_v)
            self.optimizer_q = self.accelerator.prepare(self.optimizer_q)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer_v", "lr_scheduler_v", "optimizer_q", "lr_scheduler_q"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)
            # --- FIX(set_epoch): our custom DataLoader has no set_epoch, so the guard above
            # never fires -> the chunk dataset + token-budget sampler stayed frozen at epoch 0
            # (warmup filter permanent, batch order identical every epoch). Propagate the epoch
            # explicitly: re-chunk the dataset, refresh the sampler's lengths/warmup_mask from the
            # new chunks, then rebuild this epoch's (reshuffled, warmup-expired) batches.
            _ds = getattr(self, "train_dataset", None)
            _bs = getattr(epoch_dataloader, "batch_sampler", None)
            if _ds is not None and hasattr(_ds, "set_epoch"):
                _ds.set_epoch(int(epoch))
                if _bs is not None and hasattr(_bs, "set_epoch"):
                    if hasattr(_ds, "sample_lens"):
                        _bs.lengths = _ds.sample_lens
                    if hasattr(_ds, "has_last"):
                        _bs.warmup_mask = _ds.has_last
                    _bs.set_epoch(int(epoch))
                    if self.is_world_process_zero():
                        print(f"[FIX set_epoch] epoch={epoch} chunks={len(_ds)} "
                              f"rank_batches={len(_bs._batches_rank)} "
                              f"warmup={'on' if _bs.epoch < _bs.warmup_epochs else 'off'}", flush=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                # Transformers 4.55+ expects this field during Trainer.training_step.
                # This custom loop predates that Trainer change and must expose the
                # actual accumulation count for the final partial update as well.
                self.current_gradient_accumulation_steps = len(batch_samples)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # FIX(DDP): all-reduce grads across ranks BEFORE clipping/step (DDP's own
                        # hooks are bypassed because we forward through model.module custom methods).
                        self._allreduce_grads(model)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                # _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                if self._stage == 'boost_q':
                                    _grad_norm = self.optimizer_q.clip_master_grads(args.max_grad_norm)
                                else:
                                    _grad_norm = self.optimizer_v.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                if self._stage == "boost_q":
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer_q),
                                    args.max_grad_norm,
                                )
                                else:
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer_v),
                                        args.max_grad_norm,
                                    )

                            else:
                                # _grad_norm = self.accelerator.clip_grad_norm_(
                                #     model.parameters(),
                                #     args.max_grad_norm,
                                # )
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )


                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                        if self._stage == "boost_q":
                            optimizer = self.optimizer_q
                            lr_scheduler = self.lr_scheduler_q
                        else:
                            optimizer = self.optimizer_v
                            lr_scheduler = self.lr_scheduler_v
                        
                        optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            # if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            #     self.lr_scheduler.step()
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                lr_scheduler.step()


                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return
        raise NotImplementedError

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler_v.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_V_NAME), weights_only=True)
                    )
                    self.lr_scheduler_q.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_Q_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_V_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_V_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_V_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_V_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_Q_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_Q_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_Q_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_Q_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, f"rank*-of-{self.args.world_size}-{OPTIMIZER_V_NAME}"))
            if self.is_fsdp_xla_v1_enabled
            else checkpoint_file_exists
        )
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, f"rank*-of-{self.args.world_size}-{OPTIMIZER_Q_NAME}"))
            if self.is_fsdp_xla_v1_enabled
            else checkpoint_file_exists
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_V_NAME)) and os.path.isfile(
            os.path.join(checkpoint, SCHEDULER_Q_NAME)
        ):
            # Load in optimizer and scheduler states
            if is_torch_xla_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                if self.is_fsdp_xla_v1_enabled:
                    check_torch_load_is_safe()
                    optimizer_v_state = torch.load(
                        os.path.join(
                            checkpoint, f"rank{self.args.process_index}-of-{self.args.world_size}-{OPTIMIZER_V_NAME}"
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    # We only need `optimizer` when resuming from checkpoint
                    optimizer_v_state = optimizer_v_state["optimizer"]
                else:
                    check_torch_load_is_safe()
                    optimizer_v_state = torch.load(
                        os.path.join(checkpoint, OPTIMIZER_V_NAME), map_location="cpu", weights_only=True
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    lr_scheduler_v_state = torch.load(
                        os.path.join(checkpoint, SCHEDULER_V_NAME), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_v_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_v_state, self.args.device)

                self.optimizer_v.load_state_dict(optimizer_v_state)
                self.lr_scheduler_v.load_state_dict(lr_scheduler_v_state)
                
                
                if self.is_fsdp_xla_v1_enabled:
                    check_torch_load_is_safe()
                    optimizer_q_state = torch.load(
                        os.path.join(
                            checkpoint, f"rank{self.args.process_index}-of-{self.args.world_size}-{OPTIMIZER_Q_NAME}"
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    # We only need `optimizer` when resuming from checkpoint
                    optimizer_q_state = optimizer_q_state["optimizer"]
                else:
                    check_torch_load_is_safe()
                    optimizer_q_state = torch.load(
                        os.path.join(checkpoint, OPTIMIZER_Q_NAME), map_location="cpu", weights_only=True
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    lr_scheduler_q_state = torch.load(
                        os.path.join(checkpoint, SCHEDULER_Q_NAME), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)
                xm.send_cpu_data_to_device(optimizer_q_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_q_state, self.args.device)
                self.optimizer_q.load_state_dict(optimizer_q_state)
                self.lr_scheduler_q.load_state_dict(lr_scheduler_q_state)
            else:
                if is_sagemaker_mp_enabled():
                    raise NotImplementedError(
                        "Loading optimizer and scheduler states is not yet implemented for SageMaker MP.")
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        load_fsdp_optimizer(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.optimizer_v,
                            self.model,
                            checkpoint,
                            **_get_fsdp_ckpt_kwargs(),
                        )
                    else:
                        check_torch_load_is_safe()
                        self.optimizer.load_state_dict(
                            torch.load(
                                os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True
                            )
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)



# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train/alfworld/alfworld_sft.json")
    parser.add_argument("--failed_data_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--task", type=str, default="alfworld")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--use_aux_reward", dest="use_aux_reward", action="store_true", default=True,
                        help="apply retrospective auxiliary-reward rules in memory (default)")
    parser.add_argument("--no_aux_reward", dest="use_aux_reward", action="store_false",
                        help="disable auxiliary reward (env reward only)")

    parser.add_argument("--cache_dir", type=str, default=None,
                        help="optional: cache the preprocessed dataset to/from this dir. "
                             "Default None = build in memory, no cache file.")
    parser.add_argument("--preprocess_only", action="store_true")

    parser.add_argument("--K", type=int, default=4)  # you want K=4
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)  # traj-level batch, usually 1-2 then grad_accum
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="token_qv_traj")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_done_head", action="store_true")
    parser.add_argument("--max_tokens_per_device", type=int, default=65536)  # dynamic token budget per device
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--stage", type=str, default="IQL", choices=["IQL", "Inference_QV", "Policy"])  # boost_v or boost_q
    parser.add_argument("--inference_output_path", type=str, default="traj_qv_inference.json")  # only for Inference_QV stage
    args = parser.parse_args()

    # Pin each DDP rank to its own GPU early (prevents the 4-GPU "devices=0" CUDA init crash under torchrun)
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if _local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(_local_rank)

    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.replace("PLACEHOLDER", args.task)
        args.cache_dir = os.path.join(args.cache_dir, 'w_assigned_rewards' if args.use_aux_reward else 'wo_assigned_rewards')
        args.cache_dir = os.path.join(args.cache_dir, args.exp_name)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    
    # 1) build trajectory dataset (in-memory; optionally cached to/from --cache_dir)
    if args.cache_dir is not None and os.path.exists(args.cache_dir):
        hf_ds = load_from_disk(args.cache_dir)
    else:
        hf_ds = build_traj_dataset(
            task=args.task,
            data_path=args.data_path,
            model_name=args.model_name,
            failed_data_path=args.failed_data_path,
            out_dir=args.cache_dir,  # None -> not written to disk
            gamma=args.gamma,
            max_length=args.max_length,
            use_aux_reward=args.use_aux_reward,
        )
    if args.preprocess_only:
        raise SystemExit

    print("[Load] trajectories:", len(hf_ds))
    # print token length stats, step stats
    token_lens = [len(item["input_ids"]) for item in hf_ds]
    step_lens = [item["ep_len"] for item in hf_ds]
    print(f"  token len: min {min(token_lens)}, max {max(token_lens)}, avg {sum(token_lens)/len(token_lens):.1f}")
    print(f"  step len: min {min(step_lens)}, max {max(step_lens)}, avg {sum(step_lens)/len(step_lens):.1f}")   
    
    
    all_weights = [w for item in hf_ds for w in item["weight"]]
    print("weight stats: min {:.4f}, max {:.4f}, p25 {:.4f}, p50 {:.4f}, p75 {:.4f}, p90 {:.4f}".format(
        min(all_weights),
        max(all_weights),
        np.percentile([all_weights], 25),
        np.percentile([all_weights], 50),
        np.percentile([all_weights], 75),
        np.percentile([all_weights], 90),
    ))
    
    all_rewards = [r for item in hf_ds for r in item["reward"]]
    print("reward stats: min {:.4f}, max {:.4f}, p25 {:.4f}, p50 {:.4f}, p75 {:.4f}, p90 {:.4f}, avg {:.4f}".format(
        min(all_rewards),
        max(all_rewards),
        np.percentile([all_rewards], 25),
        np.percentile([all_rewards], 50),
        np.percentile([all_rewards], 75),
        np.percentile([all_rewards], 90),
        sum(all_rewards) / len(hf_ds),
    ))
    
    final_rewards = [item["reward"][-1] for item in hf_ds]
    print("final reward stats: min {:.4f}, max {:.4f}, avg {:.4f}".format(
        min(final_rewards),
        max(final_rewards),
        sum(final_rewards) / len(hf_ds),
    ))
    # 3) build trajectory-step dataset (K=4)
    train_ds = TrajChunkDataset(hf_ds, K=4, max_length=args.max_length, seed=42, pad_last=True)


    # 4) model
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    model = ProbeModel(
        args.model_name,
        lora=lora,
        max_length=args.max_length,
        use_done_head=args.use_done_head,
    )
    
    
    

    if args.load_checkpoint is not None:
        # resolve checkpoint dir: pick latest checkpoint-* if needed
        if not os.path.exists(os.path.join(args.load_checkpoint, "model.safetensors")):
            ckpts = sorted(glob.glob(os.path.join(args.load_checkpoint, "checkpoint-*")))
            if ckpts:
                cand = ckpts[-1]
                if os.path.exists(os.path.join(cand, "model.safetensors")):
                    args.load_checkpoint = cand

        model_path = os.path.join(args.load_checkpoint, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.safetensors not found: {model_path}")

        # instantiate on CPU first to avoid VRAM spike


        # load state_dict (CPU)
        strict = True
        from safetensors.torch import load_file as safe_load_file
                
        state_dict = safe_load_file(model_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        del state_dict

        if missing or unexpected:
            print(f"[load_state_dict] strict={strict}")
            if missing:
                print("[load_state_dict] missing keys:", missing[:20], "..." if len(missing) > 20 else "")
            if unexpected:
                print("[load_state_dict] unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")



    print_trainable_parameters(model)

    # 5) collator
    collator = TrajStepCollator(model.tokenizer) #, max_length=args.max_length)


    # 6) train args
    cfg = SFTConfig(
        output_dir=os.path.join(args.output_dir, args.model_name.split("/")[-1]),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs + args.warmup_epochs,
        logging_steps=1,
        # save_steps=200,
        # Patched for the 15h local run: with save_strategy="no" a crash at hour 14
        # loses everything. Keep 2 rotating checkpoints (~13.5GB each).
        save_strategy="steps",
        save_steps=int(os.environ.get("CKPT_STEPS", "1000")),
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=["wandb"] if args.wandb else [],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_grad_norm=1.0,              # ✅ 必加，先用 1.0；还炸就 0.5
        warmup_ratio=0.03,              # ✅ 建议
        lr_scheduler_type="cosine",     # ✅ 建议
        optim="adamw_torch_fused",      # ✅ H100 推荐（可选）
        # alternating boost_q/expectile_v stages leave half the params gradient-free
        # each step, and PEFT switches adapters -> DDP must tolerate unused params.
        ddp_find_unused_parameters=True,
    )

    trainer = QTrainer(
        model=model,
        processing_class=model.tokenizer,
        args=cfg,
        train_dataset=train_ds,
        data_collator=collator,
        gamma=args.gamma,
        tau=args.tau,
        use_done_head=args.use_done_head,
        max_tokens_per_device=args.max_tokens_per_device,
        warmup_epochs=args.warmup_epochs,
    )

    # trainer.save_model()   # 去掉训练前的无谓存盘(存的是未训练模型)
    trainer.train()
    trainer.save_model()      # 只在最后存一次最终模型(顶层 model.safetensors，inf_critic 优先读它)
