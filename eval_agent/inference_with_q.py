import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils import State
import re
import copy

from peft import LoraConfig, get_peft_model, PeftModel, TaskType  # <-- import PeftModel
from safetensors.torch import load_file
from eval_agent.data_utils import rank0_print, ChatTemplate, get_chat_template

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizer
import argparse

logger = logging.getLogger("agent_frame")

N_DEBUG_EXAMPLES = 5

import random


import os
import torch
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
):
    """
    适用于：Trainer 保存的整模型 checkpoint
      - ckpt_dir/model.safetensors  (full state_dict)
    你要求：先实例化 ProbeModel，然后 load 参数
    """
    from QEvolve.train_critic import ProbeModel  # 确保是你训练用的同一个类

    # 1) 用训练时相同的 LoRAConfig 先把“参数结构”建出来（很关键）
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="FEATURE_EXTRACTION",
    )

    # 2) 先在 CPU 实例化（避免一上来占满显存）
    model = ProbeModel(base_model_name_or_path, lora, max_length=max_length, use_done_head=False)
    model.eval()

    # 3) CPU 上加载 safetensors
    model_path = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.safetensors not found: {model_path}")

    state_dict = safe_load_file(model_path)  # CPU dict
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    del state_dict

    if (missing or unexpected):
        print("[load_state_dict] missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        print("[load_state_dict] unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    # 4) 再搬到 GPU + bf16
    model = model.to(device=device)
    model.eval()

    # 5) tokenizer 一定存在（你的 ProbeModel.__init__ 里应当有）
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("ProbeModel instance has no tokenizer. Check your ProbeModel.__init__")

    return model, tokenizer

@torch.no_grad()
def score_actions_with_probe(adv_net, input_ids, attention_mask):
    """
    input_ids: [B,T], attention_mask: [B,T]
    返回:
      Q1_hat, Q2_hat: [B]
      Q_hat: [B]
    这里把 action_pos 定义为 “每条序列的最后一个非 pad token”
    这与你训练时 action_end = len(prefix)-1 的定义一致（前提：你的 prompt 最后确实以 assistant 输出结尾）
    """
    adv_net.eval()
    inner = adv_net.module if hasattr(adv_net, "module") else adv_net

    # [B] 最后一个有效 token index
    action_pos = attention_mask.long().sum(dim=1) - 1
    action_pos = action_pos.clamp_min(0).unsqueeze(1)  # [B,1] => K=1

    out = inner.forward_q_multi(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_pos=action_pos,
    )
    Q1 = out["Q1"].squeeze(1).squeeze(-1).float()  # [B]
    Q2 = out["Q2"].squeeze(1).squeeze(-1).float()  # [B]
    Q  = torch.minimum(Q1, Q2)
    # Q = (Q1 + Q2) / 2
    return Q1, Q2, Q


def parse_action(x):
    if not x:
        print('Find a empty action.')
        return False, None
    try:
        x = x.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, x)[0]
        if action:
            return True, action
        else:
            print(f'Find an invalid action: {action}')
            return False, None
    except:
        return False, None


def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
    model_path: str,
) -> Dict:
    chat:ChatTemplate = get_chat_template(model_path)
    roles = {"human": "user", "gpt": "assistant"}
    roles_list = ["user", "assistant"]

    conversations = []

    for source in sources:
        # messages = []
        # for j, sentence in enumerate(source):
        #     # role = roles[sentence["from"]]
        #     assert role == roles_list[j % 2], f"{j}"
        #     messages.append(
        #         {"role": role, "content": sentence["value"]}
        #     )
        conversations.append(chat.get_prompt(source))



    inputs = tokenizer(
        conversations,
        return_tensors="pt",
        padding=True,
        # padding_to_multiple_of=8,
    )

    return inputs

def debug_logging_info(msg: str, debug: bool):
    if debug:
        logger.info(msg)
    return

def trunc(conv):
    if len(conv) > 50: 
        # import ipdb; ipdb.set_trace()
        history_prompt = conv[:2] +  [conv[12]] + conv[-40:]
        logger.info("Truncated history for long context.")
        # logger.info("".join(history_prompt))
    else:
        history_prompt = conv
    return history_prompt

def trunc_for_adv(conv):
    if len(conv) > 50: 
        # import ipdb; ipdb.set_trace()
        history_prompt = conv[:2] +  [conv[12]] + conv[-40:]
        # logger.info("".join(history_prompt))
    else:
        history_prompt = conv[:2] +  conv[12:]
    return history_prompt
def interactive_loop(
    args: argparse.Namespace,
    task: tasks.Task,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
    debug: bool = False,
    adv_net= None,
    adv_tokenizer=None,
    mnl_func=None,
    device=None,
) -> State:
    debug_logging_info(f"Loading environment: {env_config['env_class']}", debug)
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    if args.force_first:
        env.icl_format = 'first'
        
    invalid = set()
    # reset the environment and set the prompt
    observation, state = env.reset(args.num_icl_examples)


    init_msg = observation

    debug_logging_info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}", debug)

    cur_step = 0
    # import ipdb;ipdb.set_trace()
    while not state.finished:
        debug_logging_info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n", debug)
        cur_step += 1
        # agent act
        # try:
        if True:
            # history = 
            # if len(state.history_conversation) > 60: 
            #     # import ipdb; ipdb.set_trace()
            #     history_prompt = state.history_conversation[:2] +  [state.history_conversation[12]] + state.history_conversation[-40:]
            #     logger.info("Truncated history for long context.")
            #     # logger.info("".join(history_prompt))
            # else:
            #     history_prompt = state.history_conversation
            history_prompt = trunc(state.history_conversation)
                # logger.info("".join(history_prompt))
            print("="* 20, f"State {cur_step}", "=" * 20)
            print(state.history_conversation[-1]['content'])
            # llm_output = agent(history_prompt)
            # llm_output_list = [llm_output]
            # action_list = [parse_action(llm_output)]
            if 'Nothing happens' in history_prompt[-1]['content']:
                
                bad_action = parse_action(history_prompt[-2]['content'])[1]
                invalid.add(bad_action)
                print(f'Find invalid action: {bad_action}')
                # history_prompt[-1]['content'] = f"Observation: Nothing happens. The task is not completed yet. You have already tried the following invalid actions:  {', '.join(map(str, invalid))}."
                # history_prompt[-1]['content']+= "Please choose another reasonable action."
                history_prompt[-1]["content"] = (
                    "Observation: Your chosen action is invalid. The task is not completed yet.\n"
                    f"Invalid actions tried so far: {', '.join(map(str, invalid))}.\n"
                    "Please choose a different valid action and do not repeat any invalid action above."
                )

            else:
                invalid = set()


            if True:




                llm_output_list = agent(history_prompt)

            else:
                llm_output_list = []
                action_list = []
                # verify each action
                while len(llm_output_list) < args.num_samples:
                    if len(llm_output_list) == 0:
                        explore_add_prompt = ''
                    elif len(llm_output_list) == 1:
                        explore_add_prompt = f"\nPlease provide another reasonable response different from '{action_list[0]}'."
                    else:
                        explore_add_prompt =f"\nYou have given the the following answers. \n"
                        for i in range(len(action_list)):
                            explore_add_prompt += f"Answer {i+1}: '{action_list[i]}'\n"
                        explore_add_prompt += "Please provide a reasonable response different from previous answers."
                    
                    cur_state_history = copy.deepcopy(history_prompt)
                    cur_state_history[-1]['content'] += explore_add_prompt
                    llm_output = agent(cur_state_history)
                    if_valid, action = parse_action(llm_output)
                    if if_valid:
                        llm_output_list.append(llm_output)
                        action_list.append(action)
                
            # llm_output_list = [lll for lll in llm_output_list if parse_action(lll)]
            if args.adv_type == 'token_prediction':
                adv_prompt_list = []
                for llm_output in llm_output_list:
                    adv_conv = trunc_for_adv(state.history_conversation) + [{'role': 'assistant', 'content': llm_output}]
                    # adv_conv = state.history + [{'role': 'assistant', 'content': llm_output}]
                    adv_prompt_list.append(adv_conv)
                
                # tokenize and forward
                action_inputs = preprocess(adv_prompt_list, adv_tokenizer, args.advantage_base_model).to(device)

                # forward score
                Q1_hat, Q2_hat, Q_hat = score_actions_with_probe(
                    adv_net=adv_net,
                    input_ids=action_inputs["input_ids"],
                    attention_mask=action_inputs["attention_mask"],
                )

                print("="* 20, "Candidates", "=" * 20)
                for candidate, s1, s2 in zip(llm_output_list, Q1_hat, Q2_hat):
                    print(candidate, " | ", float(s1.item()), " | ", float(s2.item()), "\n")
                # action_inputs = preprocess(adv_prompt_list, adv_tokenizer, args.advantage_base_model).to(device)
                # with torch.no_grad():
                            
                #     adv_pred = adv_net(
                #         input_ids=action_inputs["input_ids"],
                #         attention_mask=action_inputs["attention_mask"],
                #         action_mask=action_inputs["attention_mask"],
                #     )
                #     Q_hat = torch.minimum(adv_pred["Q1_hat"], adv_pred["Q2_hat"])
                    # Q_hat = (adv_pred["Q1_hat"] + adv_pred["Q2_hat"])/2
                
                # max Q action
                maxQ_index = torch.argmax(Q_hat).item()
                llm_output = llm_output_list[maxQ_index]  # text
                
                
                
                # print("="* 20, "Candidates", "=" * 20)
                # for candidate, score1, score2 in zip(llm_output_list, adv_pred["Q1_hat"], adv_pred["Q2_hat"]):
                #     print(candidate, " | ", score1.item() , " | ", score2.item(), "\n")
                if False:
                    # Q_hat: [N] 或 [N,1]，这里先拉平
                    q = Q_hat.detach().float().view(-1)

                    # ---- normalize (z-score) ----
                    eps = 1e-6
                    q_norm = (q - q.mean()) / (q.std(unbiased=False) + eps)

                    # ---- softmax sampling ----
                    temperature = getattr(args, "q_sample_temp", 1.0)  # 可加参数控制随机性
                    probs = torch.softmax(q_norm / max(temperature, eps), dim=0)

                    # multinomial 需要 probs 是合法分布；极端情况下兜底
                    if not torch.isfinite(probs).all() or probs.sum() <= 0:
                        probs = torch.ones_like(q_norm) / q_norm.numel()

                    idx = torch.multinomial(probs, num_samples=1).item()

                    llm_output = llm_output_list[idx][0]   # text
                    chosen_gid = llm_output_list[idx][1]   # action / gid
                    chosen_Q = q[idx].item()

                    print("=" * 20, "Final Sampled", "=" * 20)
                    print(llm_output, "| action:", chosen_gid, "| Q:", chosen_Q, "| p:", probs[idx].item())

                if False:
                    from collections import defaultdict
                    key2value = defaultdict(list)

                    # llm_output_list: [(lll, gid), ...]
                    for idx_act in range(len(llm_output_list)):
                        gid = llm_output_list[idx_act][1]
                        lll = llm_output_list[idx_act][0]
                        key2value[gid].append((Q_hat[idx_act], lll))   # 存 (adv, text)

                    # group mean -> pick best group
                    best_gid = None
                    best_mean = None
                    for gid, pairs in key2value.items():
                        mean_adv = torch.stack([p[0] for p in pairs]).min()
                        if best_mean is None or mean_adv > best_mean:
                            best_mean = mean_adv
                            best_gid = gid

                    # pick best sample within best group (max Q_hat)
                    best_pair = max(key2value[best_gid], key=lambda x: x[0].item())
                    best_Q_hat, llm_output = best_pair  # llm_output 是最终选中的 lll 文本

                    # best_mean.item() 是最佳 group 的平均 advantage
                    # best_Q_hat.item() 是最终选中那条的 advantage
                    print("="* 20, "Final Chosen", "=" * 20)
                    print(llm_output, " | ", best_Q_hat.item())
                      
            elif args.adv_type == 'self_consistency':
                
                import re, random
                from collections import Counter

                # 提取每条的 Action 内容（可能为空字符串）
                acts = []
                for x in llm_output_list:
                    m = re.search(r'(?im)^[ \t]*Action\s*:\s*(.+)$', x)
                    acts.append(m.group(1).strip() if m else "")

                # 统计非空动作
                non_empty = [a for a in acts if a]

                if not non_empty:
                    # 都没匹配到 Action，直接随机一个原始输出
                    idx = random.randrange(len(llm_output_list))
                else:
                    # 多数投票；平票或全不同(计数=1)就随机选其一
                    c = Counter(non_empty)
                    maxc = max(c.values())
                    top_actions = [a for a, v in c.items() if v == maxc]
                    chosen_action = random.choice(top_actions)
                    # 在所有匹配该 action 的项里随机挑一个原始输出
                    cand_idx = [i for i, a in enumerate(acts) if a == chosen_action]
                    idx = random.choice(cand_idx)

                llm_output = llm_output_list[idx]  # 原始 LLM 输出
                # best_action = acts[idx]             # 对应的 Action

                
            elif args.adv_type == 'qlass':
                
                adv_prompt_list = []
                for llm_output in llm_output_list:
                    adv_conv = history_prompt[:3] + history_prompt[12:] + [{'role': 'assistant', 'content': llm_output[0]}]
                    # adv_conv = state.history + [{'role': 'assistant', 'content': llm_output}]
                    adv_prompt_list.append(adv_conv)
                
                # tokenize and forward
                action_inputs = preprocess(adv_prompt_list, adv_tokenizer, args.advantage_base_model).to(device)
                with torch.no_grad():
                    input_ids = action_inputs["input_ids"].to(device)
                    attention_mask = action_inputs["attention_mask"].to(device)

                    out = adv_net(input_ids, attention_mask=attention_mask)

                    if out.dim() == 3:
                        out = out.squeeze(-1)
                    q = out[:, -1].float().item()
                    
                    llm_output_index = torch.argmax(q).item()
                    llm_output = llm_output_list[llm_output_index][0]
            else:
                raise NotImplementedError
            

            debug_logging_info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n", debug)
        # except Exception as e:
        #     logger.info(f"Agent failed with error: {e}")
        #     state.success = False
        #     state.finished = True
        #     state.terminate_reason = f"Agent failed with error: {e}"
        #     break
        # environment step
        observation, state = env.step(llm_output)
        if not state.finished:
            # color the observation in blue
            debug_logging_info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n", debug)
        if state.finished:
            break

    if state.reward is not None:
        debug_logging_info(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}", debug)
    else:
        debug_logging_info(f"Task finished in {state.steps} steps. Success: {state.success}", debug)
        
    return state


def main(args: argparse.Namespace):
    # set_seed(42)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name


    if args.num_samples is not None:
        agent_config['config']['num_samples'] = args.num_samples
    output_path = os.path.join("outputs4", agent_config['config']['model_name'].replace('/', '_') + '-' + args.adv_type + args.notes, args.exp_config+args.exp_name+args.split+str(args.num_icl_examples)+'shot'+'seed-'+str(args.seed))
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        from eval.webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.eval_utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    advantage_model = None
    device = 'cuda'
    if args.adv_type == 'token_prediction':


        advantage_model, advantage_tokenizer = load_probe_full_state_dict(
            base_model_name_or_path=args.advantage_base_model,
            ckpt_dir=args.advantage_model_path,
            device=device,
            dtype=torch.bfloat16,
            max_length=4096,
            strict=True,
        )
        print("✅ adv model loaded:", type(advantage_model), "dtype:", next(advantage_model.parameters()).dtype)
    elif args.adv_type == 'qlass':
        from eval_agent.q_network import QNet_V1 as QNet
        from types import SimpleNamespace

        def load_qnet_and_tokenizer(
            qnet_path: str,
            *,
            device: str = "cuda",
            dtype: torch.dtype = torch.bfloat16,
            max_length: int = 4096,
            use_flash_attn: bool = True,
        ):
            # QNet.from_pretrained 里你原脚本用到了 args.low_cpu_mem_usage / args.use_flash_attn / args.model_name_or_path
            qnet_args = SimpleNamespace()
            qnet_args.model_name_or_path = qnet_path
            qnet_args.low_cpu_mem_usage = False
            qnet_args.use_flash_attn = use_flash_attn
            qnet_args.apply_sigmoid = False

            qnet = QNet.from_pretrained(qnet_path, None, qnet_args)
            # qnet = AutoModelForCausalLM.from_pretrained(
            #     qnet_path,
            #     torch_dtype=dtype,
            #     # low_cpu_mem_usage=True,
            #     trust_remote_code=True,
            #     device_map={"": device},
            #     # device=device,
            # )
            qnet = qnet.to(device=device)
            qnet.eval()

            tok = AutoTokenizer.from_pretrained(qnet_path, model_max_length=max_length, use_fast=False)
            # 跟你原脚本一致的 pad 处理（你们这套 preprocess 里 padding=True）
            if tok.pad_token_id is None:
                tok.pad_token = tok.unk_token if tok.unk_token_id is not None else tok.eos_token
            return qnet, tok
        
        
        advantage_model, advantage_tokenizer  = load_qnet_and_tokenizer(
            qnet_path=args.advantage_model_path,
            device=device,
            dtype=torch.bfloat16,
            max_length=4096,
            use_flash_attn=True,
        )

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # import torch, torch.nn.functional as F
    # tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
    # mdl = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").eval()

    # def nli_score(premise, hypothesis):
    #     x = tok(premise, hypothesis, return_tensors="pt", truncation=True)
    #     with torch.no_grad():
    #         logits = mdl(**x).logits[0]
    #     probs = F.softmax(logits, dim=-1)  # [contradiction, neutral, entailment]
    #     c, n, e = probs.tolist()
    #     return e - c, {"entail": e, "neutral": n, "contradict": c}
    # # 用法：nli_score(Thought, Action)

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")


    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        return

    # # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks
    n = 0
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                args, task, agent, env_config, args.debug, advantage_model, advantage_tokenizer, device=device # nli_score
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
            n+=1
            if args.debug and n>=N_DEBUG_EXAMPLES:
                break
        pbar.close()
    
    logger.warning("All tasks done.")
    # logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(int(state.success==1.))
    print(f"all traj len is {len(state_list)}")
    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Average rew==1: {sum([int(r==1) for r in reward_list])/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="webshop",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--force_first",
        action="store_true",
        help="Whether to use first in icl_format.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=1,
        help="Number of ICL examples in evaluation prompt.")
    parser.add_argument(
        "--notes",
        type=str,
        default='none',
        help="evaluation setting notes")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed.")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="")
    parser.add_argument("--adv_type", type=str, default="token_prediction", help="Type of advantage model")
    
    parser.add_argument("--advantage_base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Path to the advantage model")
    parser.add_argument("--advantage_model_path", type=str, default="/scratch-shared/yzhang2/QLASS-Adv/token_adv-v10-large_lr_pairwise-pref_only-p30failed/Llama-2-7b-chat-hf/checkpoint-2100/", help="Path to the advantage model")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)