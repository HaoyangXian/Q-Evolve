#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    expert = json.loads(Path(args.expert).read_text())
    rollout = json.loads(Path(args.rollout).read_text())
    out = []
    for row in expert:
        row = dict(row)
        row.setdefault("source", "expert")
        reward = float(row.get("reward", 1.0) or 0.0)
        row.setdefault("reward", reward)
        row.setdefault("meta", {"reward": reward, "success": reward > 0, "finished": True})
        out.append(row)
    out.extend(rollout)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
