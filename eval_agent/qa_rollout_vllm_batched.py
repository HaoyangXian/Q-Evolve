#!/usr/bin/env python3
"""vLLM batched multi-step QA rollout for Q-Evolve.

Replaces the single-question HF generate loop with:
  1. Cross-question batching via vLLM continuous batching.  Each GPU runs one
     vLLM offline engine and concurrently maintains ``--concurrency`` active
     questions; every round submits one prompt per active question as a single
     ``llm.generate`` batch.
  2. Prefix-cache reuse.  ``enable_prefix_caching=True`` lets vLLM reuse the KV
     of the already-generated conversation prefix, so each new step only
     computes attention/MLP over the newly appended tokens.  Text content is
     byte-for-byte identical to the old single-question loop.
  3. Inference tuning: CUDA graph (``enforce_eager=False``),
     ``gpu_memory_utilization`` budget, ``max_num_seqs`` == concurrency, and
     ``max_model_len`` capped to avoid OOM.

The per-step protocol (temperature, prompt, termination marker, token caps) is
unchanged from ``qa_rollout_chunked_context.py``.
"""

import argparse
import json
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vllm import LLM, SamplingParams

from eval_agent.qa_verifier import answers_equal, finalize_rollout, has_explicit_final_marker, split_steps


# ---------------------------------------------------------------------------
# data helpers (kept identical to qa_rollout_chunked_context.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# prompt / protocol (identical to qa_rollout_chunked_context.py)
# ---------------------------------------------------------------------------
INITIAL_USER = (
    "Solve the following problem one concise reasoning step at a time. "
    "Output exactly one useful step per turn. Do not repeat previous steps. "
    "On the final turn output <FINAL_ANSWER>...</FINAL_ANSWER> and stop.\n\n"
)
CONTINUE_USER = "Continue with exactly one concise new reasoning step."


def initial_messages(question):
    return [{"role": "user", "content": INITIAL_USER + question}]


def build_prompt_tokens(tokenizer, messages, context_max_tokens, step_max_new_tokens):
    """Render the message list to token ids, truncating old turns when needed.

    Mirrors ``generate_step``'s context bounding but operates on tokens so the
    prefix that survives truncation is stable (good for prefix caching).
    """
    if context_max_tokens and context_max_tokens > step_max_new_tokens:
        while len(messages) > 1:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if len(ids) + step_max_new_tokens <= context_max_tokens:
                break
            # drop the oldest assistant turn and its follow-up user prompt
            del messages[1:3]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


class ActiveQuestion:
    """Mutable per-question rollout state."""

    __slots__ = ("base_id", "question", "gold", "messages", "chunks", "done")

    def __init__(self, base_id, question, gold):
        self.base_id = base_id
        self.question = question
        self.gold = gold
        self.messages = initial_messages(question)
        self.chunks = []
        self.done = False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--step_max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--context_max_tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=6,
                    help="active questions per GPU (== max_num_seqs)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.88)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--termination_mode", choices=("strict", "cue", "consensus"),
                    default="strict")
    args = ap.parse_args()
    random.seed(args.seed)

    rows = load_rows(args.input)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[: args.limit]

    # strip rows without a question/gold up front so concurrency window is real
    tasks = []
    for idx, row in enumerate(rows):
        question, gold = get_question(row), get_gold(row)
        if not question or not gold:
            continue
        base_id = str(row.get("id", row.get("problem_idx", idx)))
        tasks.append((base_id, question, gold))

    # build engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.concurrency,
        max_model_len=args.context_max_tokens,
        enforce_eager=False,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    tokenizer = llm.get_tokenizer()

    # Stop right after the terminal XML marker.  Use the string form so vLLM
    # matches the full token *sequence* "</FINAL_ANSWER>" (the old per-token
    # stop_token_ids approach breaks "</FINAL_ANSWER>" into pieces and can
    # trigger early on a fragment like "< FINAL").
    sp = SamplingParams(
        max_tokens=args.step_max_new_tokens,
        temperature=0.0 if args.temperature <= 0 else args.temperature,
        top_p=1.0 if args.temperature <= 0 else 0.95,
        stop=["</FINAL_ANSWER>"],
        include_stop_str_in_output=True,
        skip_special_tokens=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    total_written = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        pending = [ActiveQuestion(i, q, g) for i, q, g in tasks]
        active = []          # currently generating questions (round-robin window)

        def flush(aq):
            nonlocal total_written
            raw = "\n\n".join(aq.chunks)
            normalized, fallback_pred, finished_ = finalize_rollout(raw, args.termination_mode)
            steps = split_steps(normalized)
            if len(steps) > args.max_steps:
                steps = steps[: args.max_steps]
                finished_ = False
            if finished_ and not has_explicit_final_marker("\n".join(steps)):
                finished_ = False
            pred = steps[-1] if steps else raw
            if finished_ and fallback_pred:
                pred = steps[-1]
            ok = finished_ and (answers_equal(pred, aq.gold) or answers_equal(raw, aq.gold))
            conv = make_conversations(aq.question, steps)
            rec = {
                "id": f"{aq.base_id}::sample0",
                "question": aq.question,
                "gold_answer": aq.gold,
                "pred_answer": pred,
                "assigned_reward": [0.0] * len(steps),
                "meta": {"reward": 1.0 if ok else 0.0, "success": bool(ok), "finished": bool(finished_)},
                "conversations": conv,
                "source": "rollout",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_written += 1

        while pending or active:
            # refill the window
            while len(active) < args.concurrency and pending:
                active.append(pending.pop(0))

            if not active:
                break

            # build one batch: one prompt per active question (as token ids so
            # vLLM prefix caching reuses the shared conversation prefix)
            batch_prompts = []
            for aq in active:
                batch_prompts.append(
                    {"prompt_token_ids": build_prompt_tokens(
                        tokenizer, aq.messages, args.context_max_tokens, args.step_max_new_tokens)}
                )

            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sp,
                use_tqdm=False,
            )

            still_active = []
            for aq, out in zip(active, outputs):
                text = out.outputs[0].text.strip()
                # advance exactly one step
                if not text:
                    # empty step -> treat as finished (cannot make progress)
                    aq.done = True
                    flush(aq)
                    continue
                aq.chunks.append(text)
                aq.messages.append({"role": "assistant", "content": text})
                if has_explicit_final_marker(text) or len(aq.chunks) >= args.max_steps:
                    aq.done = True
                    flush(aq)
                else:
                    aq.messages.append({"role": "user", "content": CONTINUE_USER})
                    still_active.append(aq)

            active = still_active

    print(f"wrote {total_written} trajectories to {args.output}", flush=True)


if __name__ == "__main__":
    main()
