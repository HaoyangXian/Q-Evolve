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
logger = logging.getLogger("agent_frame")

N_DEBUG_EXAMPLES = 5
from typing import Dict, Optional, Tuple

ANCHOR = "Now, here is the task.\nTask Description:\n"
ANCHOR_WEBSHOP = "Now, here is the task.\n"

def split_prompt_around_task(text: str, keep_anchor_in_task: bool = False) -> Dict[str, str]:
    """
    Split full prompt into:
      - prefix: everything before 'Now, here is the task.'
      - task_block: from that anchor (or after it) to end

    Args:
      keep_anchor_in_task: if True, task_block starts with the anchor line;
                           if False, task_block starts right after the anchor.
    """
    idx = text.lower().find(ANCHOR.lower())
    anchor_len = len(ANCHOR)
    if idx == -1:
        idx =text.lower().find(ANCHOR_WEBSHOP.lower())
        anchor_len = len(ANCHOR_WEBSHOP)
    if idx == -1:
        raise ValueError(f"Anchor not found: {ANCHOR} and {ANCHOR_WEBSHOP}")

    prefix = text[:idx].rstrip()

    if keep_anchor_in_task:
        task_block = text[idx:].lstrip()
    else:
        task_block = text[idx + anchor_len:].lstrip()

    return prefix, task_block



def debug_logging_info(msg: str, debug: bool):
    if debug:
        logger.info(msg)
    return
import re

def is_invalid_observation(msg) -> bool:
    if msg.get("role") != "user":
        return False
    c = (msg.get("content") or "").lower()
    return (
        "nothing happen" in c
        or "is invalid" in c
        or "invalid action" in c
        or "Error Input" in c
    )

def prune_invalid_dialogues(convs):
    """
    删除所有 invalid 的对话片段：
    - 如果遇到 invalid observation（user），就把它删掉
    - 同时把它前一条 assistant action 也删掉（如果存在）
    """
    keep = []
    for i, msg in enumerate(convs):
        
        if i != 0 and is_invalid_observation(msg):
            # drop previous assistant action if it is right before this obs
            if keep and keep[-1].get("role") == "assistant":
                keep.pop()
            # drop this invalid observation
            continue
        keep.append(msg)
    return keep


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


def interactive_loop(
    args: argparse.Namespace,
    task: tasks.Task,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
    debug: bool = False,
) -> State:
    debug_logging_info(f"Loading environment: {env_config['env_class']}", debug)
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    if args.force_first:
        env.icl_format = 'first'
        
    # reset the environment and set the prompt

    instruction_from_sft = agent.set_task(task.task_id)
    # Teacher-forcing: pin the env goal to the reference instruction so replayed reference actions
    # get scored against the SAME goal (esp. price_upper) they were originally collected under.
    observation, state = env.reset(args.num_icl_examples, instruction_text=instruction_from_sft)
    init_msg = observation

    debug_logging_info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}", debug)

    cur_step = 1
    # import ipdb;ipdb.set_trace()
    while not state.finished:
        debug_logging_info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n", debug)
        cur_step += 1
        # agent act
        llm_output: str = agent(state.history)
        if not llm_output:
            print("No more action from SFT data. Ending the episode.")
            break
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

class ReproduceSFTAgent:
    def __init__(self, sft_data: List[Dict[str, Any]]):
        self.sft_data = sft_data
        self.bad_count = 0
        
    def set_task(self, task_id: str):
        self.traj = [d for d in self.sft_data if d['id'] == task_id]
        if len(self.traj) != 1:
            print("Multiple trajectory detected! Please check!")
        self.step_index = 0
        self.task_id = task_id
        self.traj = self.traj[0]['conversations']
        return self.traj[2]['value']

    def __call__(self, input_convs: List[Dict[str, str]]) -> str:
        if self.step_index >= len(self.traj): 
            # raise IndexError("No more SFT data available.")
            return None

        assert self.traj[self.step_index]['from'] == 'human', "Expecting human message."
        assert self.traj[self.step_index+1]['from'] == 'gpt', "Expecting gpt message."
        if self.step_index == 0:
            # remove task description from input_convs
            init_msg = input_convs[0]['content']
            first_obs, task_desc = split_prompt_around_task(init_msg)
            if not first_obs.strip() == self.traj[0]['value'].strip():
                self.traj[0]['value']= self.traj[0]['value'].strip().replace("    ", '')
                self.bad_count += 1
                task_desc = task_desc.replace("greenhouse", "green house")
                print(f"Corrected initial observation for task {self.task_id}. Total corrections: {self.bad_count}")
            assert first_obs.strip() == self.traj[0]['value'].strip(), "Initial observation does not match SFT data."
            
            self.step_index += 1  # move to next message
            assert self.traj[self.step_index]['from'] == 'gpt', "Expecting gpt message."
            assert self.traj[self.step_index]['value'].strip() == "OK"
            self.step_index += 1  # move to next message
            assert self.traj[self.step_index]['from'] == 'human', "Expecting human message."
            # assert self.traj[self.step_index]['value'].strip() == task_desc.strip(), "Task description does not match SFT data."
            if not self.traj[self.step_index]['value'].strip() == task_desc.strip():
                # print("Please check the following mismatch:")
                # print(f"SFT data: {self.traj[self.step_index]['value']}")
                # print(f"Input convs: {task_desc}")
                self.bad_count += 1
                print(f"Corrected initial observation for task {self.task_id}. Total corrections: {self.bad_count}")
            self.step_index += 1  # move to next message
            assert self.traj[self.step_index]['from'] == 'gpt', "Expecting gpt message."
            
            action = self.traj[self.step_index]['value']
            self.step_index += 1
            return action
        if not self.traj[self.step_index]['value'].strip() == input_convs[-1]['content'].strip():
            # print("Please check the following mismatch:")
            # print(f"SFT data: {self.traj[self.step_index]['value']}")
            # print(f"Input convs: {input_convs[-1]['content']}")
            self.bad_count += 1
            print(f"Corrected initial observation for task {self.task_id}. Total corrections: {self.bad_count}")
        self.step_index += 1
        action = self.traj[self.step_index]['value']
        self.step_index += 1
        return action
    

def main(args: argparse.Namespace):
    # set_seed(42)
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)

    # data path: data/train/env_name/env_name_sft.json
  
    with open(os.path.join("data", "train", args.exp_config, f"{args.exp_config}_sft.json")) as f:
        sft_data = json.load(f)
    
    
    agent = ReproduceSFTAgent(sft_data)

    output_path = os.path.join("outputs4", f"reproduced_{args.exp_config}_sft")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        from envs.webshop.src.webshop.web_agent_site.envs import WebAgentTextEnv
        # from eval.webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True, seed=args.seed, teacher_force=(os.environ.get('WEBSHOP_TF','1')=='1'))
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.eval_utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    
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


    
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    

    # if len(done_task_id) == n_tasks:
    #     logger.info("All tasks done. Exiting.")
    #     return

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
                args, task, agent, env_config, args.debug
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
        default=0,
        help="Number of ICL examples in evaluation prompt.")
    parser.add_argument(
        "--notes",
        type=str,
        default='top_k=-1',
        help="evaluation setting notes")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)