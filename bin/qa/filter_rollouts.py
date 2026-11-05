#!/usr/bin/env python3
"""Keep only complete strict-termination rollout trajectories."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    total = kept = 0
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open(encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = row.get("meta") or {}
            conv = row.get("conversations") or []
            gpt = [x.get("value", "") for x in conv if x.get("from") == "gpt"]
            last = gpt[-1] if gpt else ""
            full_text = "\n".join(gpt)
            complete = (
                meta.get("finished") is True
                and len(row.get("assigned_reward") or []) > 0
                and full_text.count("<FINAL_ANSWER>") == 1
                and full_text.count("</FINAL_ANSWER>") == 1
                and "<FINAL_ANSWER>" in last
                and "</FINAL_ANSWER>" in last
                and bool(row.get("pred_answer"))
                and bool(row.get("gold_answer"))
            )
            if complete:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept}/{total} complete strict rollouts")


if __name__ == "__main__":
    main()
