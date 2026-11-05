# data.py
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch
from collections import defaultdict
from tqdm import tqdm

def parse_alfworld_conversation(item):
    conv = item["conversations"]

    task_text = ""
    obs_list, action_list = [], []

    break_flag = 0
    for idx, turn in enumerate(conv):
        if idx + 1 >= len(conv):
            break
        if idx == 0:
            assert turn["from"] == "human" and conv[idx+1]['from'] == 'gpt' and conv[idx+1]["value"].lower() == 'ok'
            task_instruction = conv[0]['value']
        elif idx % 2 == 0:
            assert turn["from"] == "human" and conv[idx+1]['from'] == 'gpt'
            if "Your task is to:" in turn['value']:
                obs, task_description = turn['value'].split("Your task is to:")
                obs = f'Observation: {obs}'
                # print(obs)
            else:
                obs = turn['value']
            obs = obs.strip()
            task_description = task_description.strip() 
            action = conv[idx+1]['value'].strip()
            obs_list.append(obs)
            action_list.append(action)
        
        if len(obs_list) >= 20:
            break_flag = 1
            break
            
    N = len(action_list)
    R     = [0.0 for _ in range(N)]
    done  = [0.0 for _ in range(N)]

    if not break_flag:
        if 'reward' in item:
            R[-1]    = int(item['reward']) 
        done[-1] = 1.0
    return dict(task_description=task_description, obs=obs_list, action=action_list, reward=R, done=done)

def parse_alfworld_conversation_from_list(conv):


    task_text = ""
    obs_list, action_list = [], []

    for idx, turn in enumerate(conv):
        if idx == 0:
            assert turn["role"] == "user" and conv[idx+1]['role'] == 'assistant' and conv[idx+1]["content"].lower() == 'ok'
            task_instruction = conv[0]['content']
        elif idx % 2 == 0:
            try:
                assert turn["role"] == "user" and conv[idx+1]['role'] == 'assistant'
            except:
                print('Assertion Error.')
                print("Assertion failed for turn:", turn)
                print('Index:', idx, 'of', len(conv))
                
                print('Roles:', turn["role"], conv[idx+1]['role'] if idx + 1 < len(conv) else None )
                print('Current conversation:')
                print(conv)
                continue    
            if "Your task is to:" in turn['content']:
                obs, task_description = turn['content'].split("Your task is to:")
                obs = f'Observation: {obs}'
                # print(obs)
            else:
                obs = turn['content']
            obs = obs.strip()
            task_description = task_description.strip() 
            action = conv[idx+1]['content'].strip()
            obs_list.append(obs)
            action_list.append(action)

    return dict(task_description=task_description, obs=obs_list, action=action_list)

class OfflineReplayBuffer:
    """
    直接接收 parse_alfworld_conversation(item) 的返回：
      {"S": List[str], "A": List[str], "R": List[List[float]], "done": List[List[float]]}
    提供 sample(batch_size) -> s, a, r, done
    """
    def __init__(self, trajectories, gamma):
        self.data = defaultdict(list)  # 仅此一个容器
        self.gamma = gamma
        self.step_count = 0
        self.ep_count = 0
        self._next_history_id = 0
        for traj in trajectories:
            self.insert_trajectory(traj)

    def insert_trajectory(self, seq):
        seq_len = len(seq['action'])

        hid = self._next_history_id
        self._next_history_id += 1  # 递增 trajectory id
        
        self.data['traj_id'].extend([hid] * seq_len)   # ✅ 每个 step 都有相同 history_id
        
        self.data['task_description'].extend([seq['task_description'] for _ in range(seq_len)])
        self.data['obs'].extend(seq['obs'])
        self.data['action'].extend(seq['action'])
        self.data['reward'].extend(seq['reward'])
        self.data['done'].extend(seq['done'])

 
        self.data['next_obs'].extend(seq['obs'][1:] + seq['obs'][-1:])
        self.data['next_action'].extend(seq['action'][1:] + seq['action'][-1:])

        # calculate return 
        G = self.compute_return(seq)
        self.data["G"].extend(G)

        self.step_count += len(seq['action'])
        self.ep_count += 1
 

    def compute_return(self, seq):
        G = []
        pre = 0.0
        for i in reversed(range(len(seq["action"]))):
            pre  = seq['reward'][i] + self.gamma * pre * (1 - seq['done'][i])
            G.append(pre)
        return G
    def __len__(self):
        return self.step_count

    def sample(self, batch_size: int):
        # 随机索引
        idx_t = torch.randint(0, self.step_count, (batch_size,), dtype=torch.long)
        ids = idx_t.tolist()

        # 组装 batch（字符串列表直接索引；数值列用张量切片）
        batch = dict(
            task_description=[self.data['task_description'][i] for i in ids],
            obs=[self.data['obs'][i] for i in ids],
            action=[self.data['action'][i] for i in ids],
            next_obs=[self.data['next_obs'][i] for i in ids],
            next_action=[self.data['next_action'][i] for i in ids],
            reward=[self.data['reward'][i] for i in ids],          
            G=[self.data['G'][i] for i in ids], 
            done=[self.data['done'][i] for i in ids], 
        )
        return batch

# ==== 单文件调试入口 ====
if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path

    parser = argparse.ArgumentParser("SFT data -> Samples")
    parser.add_argument("--path", type=str, default="data/train/alfworld/alfworld_sft.json")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"[ERR] File not found: {p}", file=sys.stderr)
        sys.exit(1)

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # data 既可能是单条 dict，也可能是 list[dict]
    items = data if isinstance(data, list) else [data]

    parsed_list = []
    total_steps = 0
    for idx, item in enumerate(items):
        parsed = parse_alfworld_conversation(item)
        parsed_list.append(parsed)
        n = len(parsed["action"])
        total_steps += n
        if n > 35:
            print(n)
        print(f"[OK] item#{idx}: steps={n}")


    print(f"\n[SUMMARY] items: {len(parsed_list)} | total steps: {total_steps}\n")

    buf = OfflineReplayBuffer(parsed_list, gamma=0.95)  # 也支持传单个 dict
    print(len(buf))                         # 总 step 数

    
    # 抽一个 batch 并打印
    batch = buf.sample(batch_size=args.batch_size)

    # 统一把数值变成 float，便于可读打印
    def _to_float(x):
        if isinstance(x, torch.Tensor):
            # 兼容 [B,1] 或标量
            return float(x.item()) if x.ndim == 0 else float(x[0].item())
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
            return float(x[0])
        return float(x)

    B = len(batch["obs"])
    for i in range(B):
        print(f"--- batch item #{i} ---")
        print("task_description:", batch["task_description"][i])
        print("obs:", batch["obs"][i])
        print("action:", batch["action"][i])
        print("next_obs:", batch["next_obs"][i])
        print("next_action:", batch["next_action"][i])

        # reward / G / done 可能是 tensor 或 list[[v]]
        if isinstance(batch["reward"], torch.Tensor):
            r_i = _to_float(batch["reward"][i])
        else:
            r_i = _to_float(batch["reward"][i])

        if batch["G"] is None:
            G_i = None
        elif isinstance(batch["G"], torch.Tensor):
            G_i = _to_float(batch["G"][i])
        else:
            G_i = _to_float(batch["G"][i])

        if isinstance(batch["done"], torch.Tensor):
            d_i = _to_float(batch["done"][i])
        else:
            d_i = _to_float(batch["done"][i])

        print(f"reward: {r_i}   G: {G_i}   done: {d_i}\n")