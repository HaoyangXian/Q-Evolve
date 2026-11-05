#!/usr/bin/env python3
"""Re-score saved QA rollouts with an explicit termination policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_agent.qa_verifier import answers_equal, finalize_rollout, split_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=("strict", "cue", "consensus"), default="cue")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open() if line.strip()]
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in rows:
            raw = "\n\n".join(
                m.get("value", "") for m in row.get("conversations", []) if m.get("from") == "gpt"
            )
            normalized, pred, finished = finalize_rollout(raw, args.mode)
            steps = split_steps(normalized)
            pred = steps[-1] if steps else pred
            ok = finished and answers_equal(pred, row.get("gold_answer", ""))
            row["pred_answer"] = pred
            row["assigned_reward"] = [0.0] * len(steps)
            row["meta"] = {"reward": 1.0 if ok else 0.0, "success": bool(ok), "finished": bool(finished)}
            row["termination_mode"] = args.mode
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
