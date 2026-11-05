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

import random
from typing import Callable, Dict, List, Optional

def _extract_webshop_task_from_prompt(text: str) -> str:
    """
    Try to extract the raw task text from a WebShop prompt.

    Expected pattern (your original code):
      ... 'here is the task.\nWebShop [SEP] Instruction: [SEP] ' + <TASK> + ' [SEP] Search' ...
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    anchor = "here is the task.\nWebShop [SEP] Instruction: [SEP] "
    if anchor in text:
        text = text.split(anchor, 1)[-1]

    # cut at the first occurrence of " [SEP] Search"
    sep_search = " [SEP] Search"
    if sep_search in text:
        text = text.split(sep_search, 1)[0]

    return text.strip()


def perturb_messages_for_webshop(
    instruction: str,
    messages: List[Dict[str, str]],
    *,
    prompt_template: str,
    paraphrase_fn: Optional[Callable[[str], str]] = None,
    seed: Optional[int] = None,
    max_chars: int = 2000,
) -> List[Dict[str, str]]:
    """
    Rewrite the task in messages[0]['content'] (WebShop prompt) by paraphrasing the task text.

    Args:
        instruction: The outer instruction used to format final prompt.
        messages: Chat messages list; will modify messages[0]['content'] in-place.
        prompt_template: A format string that must contain '{instruction}' and '{task}' placeholders.
                         e.g. prompt_template.format(instruction=..., task=...)
        paraphrase_fn: Callable that takes raw task str and returns paraphrased str.
                       If None, falls back to simple lexical perturbation (no model call).
        seed: Optional random seed for reproducibility.
        max_chars: Safety clamp for very long task strings.
        max_chars: Maximum chars fed into paraphrase_fn.

    Returns:
        The same `messages` list (mutated).
    """
    if seed is not None:
        random.seed(seed)

    if not messages or not isinstance(messages, list) or "content" not in messages[0]:
        raise ValueError("messages must be a non-empty list and messages[0] must contain 'content'.")

    original_prompt = messages[0]["content"]
    raw_task = _extract_webshop_task_from_prompt(original_prompt)

    if not raw_task:
        # If parsing fails, degrade gracefully: treat full prompt as "task"
        raw_task = original_prompt.strip()

    raw_task = raw_task[:max_chars]

    # ---- Paraphrase (pluggable) ----
    if paraphrase_fn is None:
        # No external model available: do a lightweight perturbation so pipeline still runs.
        paraphrased = _cheap_paraphrase(raw_task)
    else:
        paraphrased = paraphrase_fn(raw_task).strip()
        if not paraphrased:
            paraphrased = _cheap_paraphrase(raw_task)

    task_field = f"WebShop [SEP] Instruction: [SEP] {paraphrased} [SEP] Search"
    messages[0]["content"] = prompt_template.format(instruction=instruction, task=task_field)

    return messages


def _cheap_paraphrase(text: str) -> str:
    """
    A fallback paraphraser that makes small, deterministic-ish edits.
    This is NOT semantically guaranteed like an LLM paraphrase, but prevents missing-function crashes.
    """
    # Simple templates to add variety without changing meaning too much.
    prefixes = [
        "Please help with the following task:",
        "Your task is:",
        "Goal:",
        "Objective:",
        "Find items that satisfy:",
    ]
    # Tiny synonym map (safe, minimal)
    swaps = {
        "buy": "purchase",
        "cheap": "budget",
        "inexpensive": "budget",
        "find": "locate",
        "show": "display",
        "need": "require",
    }

    words = text.split()
    # random small swap
    for i, w in enumerate(words):
        key = w.lower().strip(",.?!;:\"'()[]{}")
        if key in swaps and random.random() < 0.15:
            words[i] = w.replace(key, swaps[key])

    rewritten = " ".join(words).strip()
    return f"{random.choice(prefixes)} {rewritten}"

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
    max_trunc_step=40,
    left_step_after_trunc=35,
) -> State:
    debug_logging_info(f"Loading environment: {env_config['env_class']}", debug)
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    if args.force_first:
        env.icl_format = 'first'
        
    # reset the environment and set the prompt
    observation, state = env.reset(args.num_icl_examples)


    init_msg = observation
    # if args.exp_config == 'webshop' and args.perturb:
    #     old_instruction, new_instruction = perturb_messages_for_webshop(copy.deepcopy(observation))
    debug_logging_info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}", debug)

    cur_step = 1
    # import ipdb;ipdb.set_trace()
    while not state.finished:
        debug_logging_info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n", debug)
        cur_step += 1
        # agent act
        try:
            # input_convs = copy.deepcopy(state.history)
            # if args.exp_config == 'webshop' and args.perturb:
            #     input_convs =  copy.deepcopy(state.history)
            #     for msg in input_convs:
            #         # replace user message content with perturbed version
            #         if msg['role'] == 'user':
            #             msg['content'] = msg['content'].replace(old_instruction, new_instruction)
                
            # else:
            input_convs =  copy.deepcopy(state.history)
            
            if len(input_convs) > 3:
                if "Nothing happen" in input_convs[-1]['content']:
                    bad_action = parse_action(input_convs[-2]['content'])[1]
                    # invalid.add(bad_action)
                    print(f'Find invalid action: {bad_action}')
                    input_convs[-1]["content"] = (
                        f"Observation: '{bad_action}' is invalid action. " # Nothing happens. The task is not completed yet.\n"
                        # f"Invalid actions tried so far: {', '.join(map(str, invalid))}.\n"
                        "The task is not complete. "
                        "Try another action to complete your task."
                    )
                elif " Error Input" in input_convs[-1]['content']:
                    print("bad input")
            # if len(input_convs) > 60:
            # input_convs = prune_invalid_dialogues(input_convs)
            if len(input_convs) > max_trunc_step:
                # keep the first two messages (system prompt and first user message)
                # and the last 18 messages
                input_convs = input_convs[:4] + input_convs[-left_step_after_trunc:]
            llm_output: str = agent(input_convs)
            if "Action" not in llm_output:
                print(f"Check: {llm_output}")
            debug_logging_info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n", debug)
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = f"Agent failed with error: {e}"
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


def main(args: argparse.Namespace):
    # set_seed(42)
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    output_path = os.path.join("outputs4", agent_config['config']['model_name'].replace('/', '_'), args.exp_config+args.exp_name+args.split+str(args.num_icl_examples)+'shot')
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
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
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
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )



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
            if args.exp_config == "webshop":
                state = interactive_loop(
                    args, task, agent, env_config, args.debug,     max_trunc_step=6, left_step_after_trunc=3,
                )
            else:
                state = interactive_loop(
                    args, task, agent, env_config, args.debug,     max_trunc_step=40, left_step_after_trunc=35,
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