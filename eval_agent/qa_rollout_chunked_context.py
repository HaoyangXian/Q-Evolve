#!/usr/bin/env python3
"""Generate multi-step QA prefix trajectories for Q-Evolve.

The script accepts JSONL or parquet records with a question and a gold answer.
It can also add reference-solution trajectories when a ``solution`` field is
available.  The output is the same alternating human/gpt envelope consumed by
``QEvolve.qa_data.preprocess_traj_dataset_qa``.
"""

import argparse
import json
import os
import random
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from eval_agent.qa_verifier import answers_equal, finalize_rollout, has_explicit_final_marker, split_steps


def load_rows(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_value(x):
    if isinstance(x, list):
        return x[0] if x else ""
    return x or ""


def get_question(row):
    return str(row.get("question", row.get("problem", "")))


def get_gold(row):
    return str(first_value(row.get("final_answer", row.get("answer", ""))))


def get_solution(row):
    solution = row.get("solution", "")
    if solution:
        return str(first_value(solution))
    # GSM8K stores the worked solution and the final ``####`` answer in its
    # answer field, so it can serve as a reference trajectory directly.
    answer = str(row.get("answer", "") or "")
    return answer if "####" in answer else ""


def make_conversations(question, steps):
    system = (
        "Solve the problem one reasoning step at a time. Each assistant turn "
        "must contain exactly one useful step. On the final turn include "
        "<FINAL_ANSWER>...</FINAL_ANSWER>."
    )
    conv = [
        {"from": "human", "value": system},
        {"from": "gpt", "value": "OK"},
        {"from": "human", "value": question},
    ]
    for i, step in enumerate(steps):
        conv.append({"from": "gpt", "value": step})
        if i + 1 < len(steps):
            conv.append({"from": "human", "value": "Continue reasoning."})
    return conv


def reference_steps(solution, gold):
    chunks = split_steps(solution)
    if not chunks:
        chunks = [solution]
    if not any("FINAL_ANSWER" in x.upper() for x in chunks):
        chunks.append(f"<FINAL_ANSWER>{gold}</FINAL_ANSWER>")
    return chunks


def model_prompt(tokenizer, question):
    messages = [{
        "role": "user",
        "content": (
            "Solve the following problem. Emit multiple blocks, each enclosed "
            "as <STEP>...</STEP>. The last block must contain the final result "
            "inside <FINAL_ANSWER>...</FINAL_ANSWER>. Stop immediately after "
            "the closing </FINAL_ANSWER> tag.\n\n" + question
        ),
    }]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"]


class FinalAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer("</FINAL_ANSWER>", add_special_tokens=False).input_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids


def _render_messages(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def generate_step(model, tokenizer, messages, max_new_tokens, temperature, context_max_tokens=0):
    # Keep the original question and recent reasoning while bounding inference context.
    if context_max_tokens and context_max_tokens > max_new_tokens:
        while len(messages) > 3:
            prompt0 = _render_messages(tokenizer, messages)
            if len(tokenizer(prompt0, add_special_tokens=False).input_ids) + max_new_tokens <= context_max_tokens:
                break
            del messages[1:3]
    prompt = _render_messages(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([FinalAnswerStoppingCriteria(tokenizer)]),
        )
    gen = out[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_iterative(model, tokenizer, question, step_max_new_tokens, max_steps, temperature, context_max_tokens=0, wall_clock_timeout=0.0):
    """Generate one short reasoning step per model call until final answer."""
    messages = [{
        "role": "user",
        "content": (
            "Solve the following problem one concise reasoning step at a time. "
            "Output exactly one useful step per turn. Do not repeat previous steps. "
            "On the final turn output <FINAL_ANSWER>...</FINAL_ANSWER> and stop.\n\n"
            + question
        ),
    }]
    chunks = []
    t_start = time.time()
    for _ in range(max_steps):
        if wall_clock_timeout > 0 and (time.time() - t_start) > wall_clock_timeout:
            break
        chunk = generate_step(model, tokenizer, messages, step_max_new_tokens, temperature, context_max_tokens)
        if not chunk:
            break
        chunks.append(chunk)
        messages.append({"role": "assistant", "content": chunk})
        if has_explicit_final_marker(chunk):
            break
        messages.append({"role": "user", "content": "Continue with exactly one concise new reasoning step."})
    return "\n\n".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=False)
    ap.add_argument("--adapter", required=False)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--step_max_new_tokens", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--context_max_tokens", type=int, default=0)
    ap.add_argument("--wall_clock_timeout", type=float, default=0.0,
                    help="Per-question wall-clock budget in seconds; 0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--termination_mode",
        choices=("strict", "cue", "consensus"),
        default="cue",
        help="How to recover completions that omit a terminal answer marker.",
    )
    ap.add_argument("--include_reference", action="store_true")
    ap.add_argument("--reference_only", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="Append to output instead of overwriting (for resume).")
    args = ap.parse_args()
    random.seed(args.seed)

    rows = load_rows(args.input)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[: args.limit]
    model = tokenizer = None
    if not args.reference_only:
        if not args.model:
            raise ValueError("--model is required unless --reference_only is set")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if getattr(args, "adapter", None):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    count = 0
    mode = "a" if args.append else "w"
    with open(args.output, mode, encoding="utf-8") as fout:
        for row_idx, row in enumerate(rows):
            question, gold = get_question(row), get_gold(row)
            if not question or not gold:
                continue
            base_id = str(row.get("id", row.get("problem_idx", row_idx)))

            if args.include_reference or args.reference_only:
                solution = get_solution(row)
                if solution:
                    steps = reference_steps(solution, gold)
                    conv = make_conversations(question, steps)
                    rec = {
                        "id": f"{base_id}::reference",
                        "question": question,
                        "gold_answer": gold,
                        "pred_answer": gold,
                        # assigned_reward is reserved for optional process
                        # rewards; the terminal answer reward lives in meta.
                        "assigned_reward": [0.0] * len(steps),
                        "meta": {"reward": 1.0, "success": True, "finished": True},
                        "conversations": conv,
                        "source": "reference",
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1

            if args.reference_only:
                continue
            for sample_idx in range(args.samples):
                raw = generate_iterative(
                    model, tokenizer, question, args.step_max_new_tokens,
                    args.max_steps, args.temperature, args.context_max_tokens,
                    wall_clock_timeout=args.wall_clock_timeout,
                )
                normalized, fallback_pred, finished = finalize_rollout(raw, args.termination_mode)
                steps = split_steps(normalized)
                if len(steps) > args.max_steps:
                    steps = steps[:args.max_steps]
                    finished = False
                if finished and not has_explicit_final_marker("\n".join(steps)):
                    finished = False
                pred = steps[-1] if steps else raw
                if finished and fallback_pred:
                    pred = steps[-1]
                ok = finished and (answers_equal(pred, gold) or answers_equal(raw, gold))
                conv = make_conversations(question, steps)
                rec = {
                    "id": f"{base_id}::sample{sample_idx}",
                    "question": question,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "assigned_reward": [0.0] * len(steps),
                    "meta": {"reward": 1.0 if ok else 0.0, "success": bool(ok), "finished": bool(finished)},
                    "conversations": conv,
                    "source": "rollout",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} trajectories to {args.output}")


if __name__ == "__main__":
    main()
