"""Prefix-MDP trajectory preprocessing for ordinary question answering.

The environment specific runners in :mod:`train_critic` all eventually produce
the same token-span representation.  This module builds that representation
from a QA rollout JSONL file, where ``conversations`` alternates human/gpt
messages and ``assigned_reward`` contains one reward per generated step.
"""

import json
import os
from typing import Optional

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from eval_agent.data_utils import get_chat_template


def _qa_prompt(tokenizer, model_name, conversations):
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    messages = []
    for i, sent in enumerate(conversations):
        role = roles[sent["from"]]
        expected = "user" if i % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError(f"QA conversation role mismatch at {i}: {role} != {expected}")
        messages.append({"role": role, "content": sent["value"]})
    return get_chat_template(model_name).get_prompt(messages)


def preprocess_traj_dataset_qa(
    data_path: str,
    failed_data_path: Optional[str],
    model_name: str,
    out_dir: Optional[str] = None,
    gamma: float = 0.95,
    max_steps: int = 128,
    max_length: int = 4096,
    use_aux_reward: bool = False,
):
    """Build the generic Q/V trajectory dataset for QA prefix MDPs.

    A QA rollout is terminal after its final generated step, including an
    incorrect answer.  This differs from ALFWorld, where a max-step failure is
    intentionally non-terminal for some legacy data.
    """
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if failed_data_path:
        with open(failed_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length
    records = []

    for item in rows:
        conv = item.get("conversations")
        if not conv:
            continue
        # The first human/gpt pair is the instruction/OK prefix; every later
        # human/gpt pair is one reasoning action.
        step_states, step_actions, step_nexts = [], [], []
        for idx in range(2, len(conv), 2):
            if idx + 1 >= len(conv):
                break
            state = conv[: idx + 1]
            action = conv[: idx + 2]
            next_state = conv[: idx + 3] if idx + 2 < len(conv) else action
            step_states.append(state)
            step_actions.append(action)
            step_nexts.append(next_state)

        if not step_states:
            continue
        if len(step_states) > max_steps:
            step_states = step_states[:max_steps]
            step_actions = step_actions[:max_steps]
            step_nexts = step_nexts[:max_steps]

        rewards = item.get("assigned_reward")
        if rewards is None:
            rewards = [0.0] * len(step_states)
        rewards = [float(x) for x in rewards[: len(step_states)]]
        if len(rewards) < len(step_states):
            rewards.extend([0.0] * (len(step_states) - len(rewards)))

        meta = item.get("meta") or {}
        final_reward = meta.get("reward", item.get("final_reward", 0.0))
        final_reward = float(final_reward or 0.0)
        final_done = 1.0
        rewards[-1] += final_reward
        won = 1.0 if final_reward > 0.0 else 0.0

        full_prompt = _qa_prompt(tokenizer, model_name, conv)
        full_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
        state_end, action_end, next_end = [], [], []
        truncated = False
        for state, action, nxt in zip(step_states, step_actions, step_nexts):
            ps = tokenizer(_qa_prompt(tokenizer, model_name, state), add_special_tokens=False).input_ids
            pa = tokenizer(_qa_prompt(tokenizer, model_name, action), add_special_tokens=False).input_ids
            pn = tokenizer(_qa_prompt(tokenizer, model_name, nxt), add_special_tokens=False).input_ids
            if len(pn) > max_length:
                truncated = True
                break
            state_end.append(len(ps) - 1)
            action_end.append(len(pa) - 1)
            next_end.append(len(pn) - 1)

        ep_len = len(state_end)
        # A partial prefix has no trustworthy terminal reward.  Dropping the
        # whole trajectory is safer than assigning its final answer reward to
        # an earlier truncated step.
        if ep_len == 0 or truncated:
            continue
        records.append({
            "input_ids": full_ids[: next_end[-1] + 1],
            "state_end": state_end,
            "action_end": action_end,
            "next_end": next_end,
            "reward": rewards[:ep_len],
            "done": [0.0] * (ep_len - 1) + [1.0],
            "weight": ((np.arange(1, ep_len + 1, dtype=np.float32) / ep_len + won) * 0.5 + 0.5).tolist(),
            "final_done": int(final_done),
            "ep_len": int(ep_len),
        })

    ds = Dataset.from_list(records)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
    return ds
