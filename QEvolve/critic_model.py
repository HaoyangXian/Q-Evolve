import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json
from datasets import Dataset
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import argparse

from trl.models import create_reference_model
import os
from collections import defaultdict

from QEvolve.data import parse_alfworld_conversation, OfflineReplayBuffer



class ProbeModel(nn.Module):
    """
    AutoModel backbone + 两个线性头（V/Q）
    在 <V>/<Q> 位置抽取最后层 hidden -> 线性回归到标量。
    LoRA 只加在 backbone 上；线性头正常训练。
    """
    def __init__(self, model_name: str, lora: LoraConfig = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        # 注册探针token
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<V>", "<Q>"]})
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        # 注入 LoRA（不冻结骨干，但只训练 LoRA 权重）
        if lora is not None:
            self.backbone.base_model.prepare_inputs_for_generation = None
            self.backbone = get_peft_model(self.backbone, lora)

        self.V_ID = self.tokenizer.convert_tokens_to_ids("<V>")
        self.Q_ID = self.tokenizer.convert_tokens_to_ids("<Q>")

        H = self.backbone.config.hidden_size
        self.head_v = nn.Linear(H, 1)
        self.head_q = nn.Linear(H, 1)

        self.config = self.backbone.config

    def forward(self, input_ids, attention_mask, v_pos=None, q_pos=None, use_cache=False, **kwargs):
        """
        与 TRL.SFTTrainer 兼容：它会把 batch 原样 **kwargs 传进来。
        我们用 v_pos / q_pos 指定 token 位置，labels_v / labels_q 监督。
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=use_cache
        )
        # last = out.hidden_states[-1]  # [B,T,H]
        last = out.last_hidden_state
        B = input_ids.size(0)
        ar = torch.arange(B, device=last.device)
        if v_pos is None:
            v_pos = torch.argmax((input_ids == self.V_ID).int(), dim=1)
        if q_pos is None:
            q_pos = torch.argmax((input_ids == self.Q_ID).int(), dim=1)

        hV = last[ar, v_pos, :]  # [B,H]
        hQ = last[ar, q_pos, :]
        # V_hat = self.head_v(hV.float())  # [B,1]
        # Q_hat = self.head_q(hQ.float())  # [B,1]
        
        V_hat = self.head_v(hV)  # [B,1]
        Q_hat = self.head_q(hQ)  # [B,1]

        return {"V_hat": V_hat, "Q_hat": Q_hat}

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # 防止 Trainer 调用时崩溃
        return args, kwargs

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "ProbeModel does not support text generation. "
            "Use backbone.generate() if you need to decode manually."
        )


@dataclass
class ProbeCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        self.V_ID = self.tokenizer.convert_tokens_to_ids("<V>")
        self.Q_ID = self.tokenizer.convert_tokens_to_ids("<Q>")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask']} for f in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        next_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            [{'input_ids': f['next_input_ids'], 'attention_mask': f['next_attention_mask']} for f in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch.update({
            "next_input_ids": next_batch['input_ids'],
            "next_attention_mask": next_batch['attention_mask'],
        })

        # ✅ 修正：原来 'labels_v' 重复，这里应该是 labels_v + labels_q + done + reward
        for k in ['labels_v', 'labels_q', 'done', 'reward']:
            batch[k] = torch.tensor([[f[k]] for f in features])

        # 取 <V>/<Q> 的首次位置（保证每条样本都有）
        ids = batch['input_ids']  # [B,T]
        v_pos = torch.argmax((ids == self.V_ID).int(), dim=1)
        q_pos = torch.argmax((ids == self.Q_ID).int(), dim=1)
        batch.update({"v_pos": v_pos, "q_pos": q_pos})

        next_ids = next_batch['input_ids']  # [B,T]
        next_v_pos = torch.argmax((next_ids == self.V_ID).int(), dim=1)
        next_q_pos = torch.argmax((next_ids == self.Q_ID).int(), dim=1)
        batch.update({"next_v_pos": next_v_pos, "next_q_pos": next_q_pos})

        return batch


def render_text(task_desc: str, obs: str, action: str) -> str:
    return f"Task: {task_desc} \nObservation: {obs.strip()} <V> \n{action.strip()} <Q>"


 
# def build_examples_from_buffer(buf, gamma: float, max_history_steps: int = 5):
#     """
#     构造每个 step 的样本，其中 text / next_text 都包含历史上下文。
#     - 支持 traj_i 分组；
#     - 限制 max_history_steps 以控制 token 长度。
#     """
#     D = buf.data
#     groups = defaultdict(list)
#     for i in range(len(D["obs"])):
#         groups[D["traj_id"][i]].append(i)

#     examples = []
#     f = float

#     for hid, idx_list in groups.items():
#         for t, i in enumerate(idx_list):
#             # ---------- 当前 step 的标签 ----------
#             yV = f(D["G"][i])
#             if i < len(D["obs"]) - 1 and D["done"][i] == 0.0:
#                 yQ = f(D["reward"][i]) + gamma * f(D["G"][i + 1])
#             else:
#                 yQ = f(D["reward"][i])

#             task_desc = D["task_description"][i]

#             # ---------- 当前 text ----------
#             start = max(0, t - max_history_steps + 1)
#             prefix = []
#             for j in idx_list[start:t]:
#                 prefix.append(
#                     f"\nHistorical Step {j - start}:\n{D['obs'][j].strip()}\n{D['action'][j].strip()}"
#                 )

#             if prefix:
#                 text = (
#                     f"Task: {task_desc}\n"
#                     + "\n".join(prefix)
#                     + f"\n\nStep for Evaluate:\n{D['obs'][i].strip()} <V>\n{D['action'][i].strip()} <Q>"
#                 )
#             else:
#                 text = (
#                     f"Task: {task_desc}\n"
#                     + "\n".join(prefix)
#                     + f"\n\nStep for Evaluate:\n{D['obs'][i].strip()} <V>\n{D['action'][i].strip()} <Q>"
#                 )

#             # ---------- 下一步 text ----------
#             if t < len(idx_list) - 1:
#                 next_start = max(0, (t + 1) - max_history_steps + 1)
#                 prefix_next = []
#                 for j in idx_list[next_start:t + 1]:  # 包含下一步
#                     prefix_next.append(
#                         f"\nHistorical Step {j - next_start}:\n{D['obs'][j].strip()}\n{D['action'][j].strip()}"
#                     )
#                 next_text = (
#                     f"Task: {task_desc}\n"
#                     + "\n".join(prefix_next)
#                     + f"\n\nStep for Evaluate:\n{D['next_obs'][i].strip()} <V>\n{D['next_action'][i].strip()} <Q>"
#                 )
#             else:
#                 # 最后一步：next_text 保留当前观察 + next_obs（通常是终止）
#                 next_text = (
#                     f"Task: {task_desc}\n"
#                     f"Step for Evaluate:\n{D['next_obs'][i].strip()} <V>\n{D['next_action'][i].strip()} <Q>"
#                 )

#             examples.append({
#                 "text": text,
#                 "next_text": next_text,
#                 "labels_v": yV,
#                 "labels_q": yQ,
#                 "done": D["done"][i],
#                 "reward": D["reward"][i],
#                 "traj_id": hid,
#             })

#     return examples


def build_history_for_each_step(D, max_history_steps: int = 5):
    """
    构造每个 step 的样本，其中 text / next_text 都包含历史上下文。
    """
    examples = []
    length = len(D['obs'])
    for t in range(length):
        task_desc = D["task_description"]

        start = max(0, t - max_history_steps + 1)
        prefix = []
        for j in range(start, t):
            prefix.append(
                f"\nHistorical Step {j - start}:\n{D['obs'][j].strip()}\n{D['action'][j].strip()}"
            )

        if prefix:
            text = (
                f"Task: {task_desc}\n"
                + "\n".join(prefix)
                + f"\n\nStep for Evaluate:\n{D['obs'][t].strip()} <V>\n"
            )
        else:
            text = (
                f"Task: {task_desc}\n"
                + "\n".join(prefix)
                + f"\n\nStep for Evaluate:\n{D['obs'][t].strip()} <V>\n"
            )

        examples.append(text)

    return examples


def build_history_for_single_step(D, max_history_steps: int = 5):
    """
    只取最后一步的 history prompt（目前没在 main 里用到）
    """
    length = len(D['obs'])
    t = length - 1
    task_desc = D["task_description"]

    start = max(0, t - max_history_steps + 1)
    prefix = []
    for j in range(start, t):
        prefix.append(
            f"\nHistorical Step {j - start}:\n{D['obs'][j].strip()}\n{D['action'][j].strip()}"
        )

    if prefix:
        text = (
            f"Task: {task_desc}\n"
            + "\n".join(prefix)
            + f"\n\nStep for Evaluate:\n{D['obs'][t].strip()} <V>\n"
        )
    else:
        text = (
            f"Task: {task_desc}\n"
            + "\n".join(prefix)
            + f"\n\nStep for Evaluate:\n{D['obs'][t].strip()} <V>\n"
        )

    return text


class Trainer(SFTTrainer):
    """
    自定义 Trainer：实现 V/Q 的 TD 目标 + delayed_model soft update
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delayed_model = create_reference_model(self.model)  # target 网络
        self._gamma = 0.95
        self._tau = 0.7
        self.compute_loss = self.compute_loss_iql

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "attention_mask", "completion_mask"]
        self._signature_columns += self.train_dataset.column_names

    def compute_loss_sarsa(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "eval" if self.control.should_evaluate else "train"

        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            v_pos=inputs['v_pos'],
            q_pos=inputs['q_pos'],
        )

        labels_v = inputs['labels_v']
        with torch.no_grad():
            delayed_outputs = self.delayed_model(
                input_ids=inputs['next_input_ids'],
                attention_mask=inputs['next_attention_mask'],
                v_pos=inputs['next_v_pos'],
                q_pos=inputs['next_q_pos'],
            )
            labels_q = inputs['reward'] + (1 - inputs['done']) * self._gamma * delayed_outputs['Q_hat']

        loss = (
            nn.functional.mse_loss(outputs['V_hat'], labels_v)
            + nn.functional.mse_loss(outputs['Q_hat'], labels_q)
        )

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(
                    inputs["attention_mask"].sum()
                ).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(
                    inputs["position_ids"].size(1),
                    device=inputs["position_ids"].device,
                )
                num_tokens_in_batch = self.accelerator.gather_for_metrics(
                    local_num_tokens
                ).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Expectile regression loss (L2^tau).
        diff = Q(s,a) - V(s)
        L = |tau - 1_{diff < 0}| * diff^2
        """
        weight = torch.where(diff < 0, 1.0 - tau, tau)
        return (weight * diff.pow(2)).mean()




    def compute_loss_iql(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ----- forward on current (s,a) -----
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            v_pos=inputs["v_pos"],
            q_pos=inputs["q_pos"],
        )
        V = out["V_hat"]  # [B,1]
        Q = out["Q_hat"]  # [B,1]

        with torch.no_grad():
            # ----- target Q(s,a) for training V -----
            # 用 target 网络算 Q(s,a)，更稳定；也符合“只用数据动作”
            tgt_sa = self.delayed_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                v_pos=inputs["v_pos"],
                q_pos=inputs["q_pos"],
            )
            Q_tgt = tgt_sa["Q_hat"]  # [B,1]

            # ----- target V(s') for TD backup -----
            tgt_next = self.delayed_model(
                input_ids=inputs["next_input_ids"],
                attention_mask=inputs["next_attention_mask"],
                v_pos=inputs["next_v_pos"],
                q_pos=inputs["next_q_pos"],
            )
            V_next_tgt = tgt_next["V_hat"]  # [B,1]

            # TD target: r + gamma * V(s') (1-done)
            q_target = inputs["reward"] + (1.0 - inputs["done"]) * self._gamma * V_next_tgt

        # (5) V loss: expectile regression of Q(s,a) onto V(s)
        v_loss = self.expectile_loss(Q_tgt - V, tau=self._tau)

        # (6) Q loss: MSE TD
        q_loss = torch.nn.functional.mse_loss(Q, q_target)

        loss = v_loss + q_loss

        # ----- (optional) keep your token averaging logic unchanged -----
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, out) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )
        if self.state.global_step % 50 == 0:
            self.soft_update(tau=0.01)

    @torch.no_grad()
    def soft_update(self, tau=0.01):
        """
        LoRA-aware Polyak averaging:
        只更新 delayed_model 中的可训练参数（LoRA）
        """
        model_state = dict(self.model.named_parameters())
        target_state = dict(self.delayed_model.named_parameters())

        for name, param in model_state.items():
            if name not in target_state:
                continue
            if param.requires_grad:
                target_param = target_state[name]
                target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)

        # del self.delayed_model
        # self.delayed_model = create_reference_model(self.model)  # target 网络


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe training with LoRA")
    parser.add_argument("--data_path", type=str, default="data/train/alfworld/alfworld_sft.json",
                        help="成功轨迹文件（JSON 数组）")
    parser.add_argument("--failed_data_path", type=str, default=None,
                        help="失败轨迹文件（JSON 数组），例如 data/train/alfworld/alfworld_failed.json")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="token_qv")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()

    # ========= 1. 加载成功 data_path =========
    # jsonl_files = [os.path.join(args.data_path,data_name) for data_name in os.listdir(args.data_path) if data_name.endswith('.jsonl') and 'combine' not in data_name]
    # data = []
    # for jsonl_file in jsonl_files:
    #     data += json.load(open(jsonl_file, "r", encoding="utf-8"))
    
    data = json.load(open(args.data_path, "r", encoding="utf-8"))
    success_items = data if isinstance(data, list) else [data]
    print(f"[INFO] 轨迹条数: {len(success_items)} 来自 {args.data_path}")

    failed_items = []
    fail_ids = set()

    # ========= 2. 如果有 failed_data_path，就按 id 覆盖 success =========
    if args.failed_data_path is not None and os.path.exists(args.failed_data_path):
        failed_data = json.load(open(args.failed_data_path, "r", encoding="utf-8"))
        failed_items = failed_data if isinstance(failed_data, list) else [failed_data]
        print(f"[INFO] 失败轨迹条数: {len(failed_items)} 来自 {args.failed_data_path}")

        # 收集 fail 里所有 id（可能有重复 id，这里只是建一个集合用于过滤 success）
        for it in failed_items:
            if isinstance(it, dict) and "id" in it:
                fail_ids.add(it["id"])

        if fail_ids and False:
            # success 里如果某个 id 出现在 fail 里，则丢掉 success 的版本
            filtered_success = [
                it for it in success_items
                if not (isinstance(it, dict) and it.get("id") in fail_ids)
            ]
            dropped = len(success_items) - len(filtered_success)
            print(f"[INFO] 因为在失败集中出现，同 id 的成功轨迹丢弃了 {dropped} 条")
            success_items = filtered_success
        else:
            print("[WARN] 失败数据中没有 'id' 字段，无法按 id 覆盖 success，直接拼接")
    else:
        if args.failed_data_path:
            print(f"[WARN] 未找到失败数据文件 {args.failed_data_path}，仅使用成功数据")

    # 注意：fail 里即使有重复 id，也不去重，全部用于训练
    items = success_items + failed_items

    print(f"[INFO] 合并后总轨迹条数: {len(items)} "
          f"(成功 {len(success_items)} + 失败 {len(failed_items)})")


    # ========= 3. parse 对话 → buffer =========
    success_items = [parse_alfworld_conversation(it, True) for it in success_items]
    failed_items = [parse_alfworld_conversation(it, False) for it in failed_items]
    trajs = success_items + failed_items

    buf = OfflineReplayBuffer(trajs, gamma=args.gamma)

    # ========= 4. 模型 & LoRA =========
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
        # task_type="CAUSAL_LM",
    )
    model = ProbeModel(args.model_name, lora)
    collator = ProbeCollator(model.tokenizer)

    # ========= 5. 构造 dataset =========
    dataset = Dataset.from_list(build_examples_from_buffer(buf, args.gamma))

    # def tokenize_data(sample):
    #     """
    #     每条样本：
    #       text: 含 <V>/<Q>
    #       next_text: 下一个状态
    #       labels_v: float
    #       labels_q: float
    #       done: float
    #       reward: float
    #     """
    #     encoded = model.tokenizer(
    #         sample["text"],
    #         truncation=True,
    #         padding=False,
    #         max_length=args.max_length,
    #     )
    #     next_encoded = model.tokenizer(
    #         sample["next_text"],
    #         truncation=True,
    #         padding=False,
    #         max_length=args.max_length,
    #     )

    #     return {
    #         "input_ids": encoded["input_ids"],
    #         "attention_mask": encoded["attention_mask"],
    #         "next_input_ids": next_encoded["input_ids"],
    #         "next_attention_mask": next_encoded["attention_mask"],
    #         "text": sample["text"],
    #         "next_text": sample["next_text"],
    #         "labels_v": sample["labels_v"],
    #         "labels_q": sample["labels_q"],
    #         "done": sample["done"],
    #         "reward": sample["reward"],
    #     }

    # dataset = dataset.map(tokenize_data, batched=False)

    def tokenize_data(batch):
        # batch: Dict[str, List[...]]
        encoded = model.tokenizer(
            batch["text"],
            truncation=True,
            padding=False,          # 不在这里 pad，后面 collate 再动态 pad
            max_length=args.max_length,
        )
        next_encoded = model.tokenizer(
            batch["next_text"],
            truncation=True,
            padding=False,
            max_length=args.max_length,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "next_input_ids": next_encoded["input_ids"],
            "next_attention_mask": next_encoded["attention_mask"],
            # 保留原文（可选）
            "text": batch["text"],
            "next_text": batch["next_text"],
            # 这些本来就是 list，直接透传
            "labels_v": batch["labels_v"],
            "labels_q": batch["labels_q"],
            "done": batch["done"],
            "reward": batch["reward"],
        }

    dataset = dataset.map(
        tokenize_data,
        batched=True,
        batch_size=1024,   # 你可以调大/调小
        num_proc=8,        # 可选：多进程加速（按CPU核数/机器情况调）
    )

    # ========= 6. TRL SFT Trainer =========
    cfg = SFTConfig(
        output_dir=os.path.join(args.output_dir, args.model_name.split('/')[-1]),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=100,
        save_steps=1000,
        bf16=torch.cuda.is_available(),
        report_to=["wandb"] if args.wandb else [],
    )

    trainer = Trainer(
        model=model,
        processing_class=model.tokenizer,
        args=cfg,
        train_dataset=dataset,
        data_collator=collator,
        # judger=args.model_name,  # 你原来就有这个 kwarg，如果没用到可以删掉
    )
    trainer.train()
