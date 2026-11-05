#!/usr/bin/env python3
"""Re-score existing QA rollouts after verifier changes."""
import argparse
import json

from eval_agent.qa_verifier import answers_equal, has_final_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    n = good = 0
    with open(args.input, encoding="utf-8") as src, open(args.output, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations") or []
            raw = conversations[-1].get("value", "") if conversations else row.get("pred_answer", "")
            finished = has_final_answer(raw) or row.get("source") == "reference"
            ok = finished and answers_equal(row.get("pred_answer", ""), row.get("gold_answer", ""))
            meta = dict(row.get("meta") or {})
            meta.update(reward=1.0 if ok else 0.0, success=bool(ok), finished=bool(finished))
            row["meta"] = meta
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            good += int(ok)
    print(f"wrote {n} rows; correct={good}")


if __name__ == "__main__":
    main()
