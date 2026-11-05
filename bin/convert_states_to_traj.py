#!/usr/bin/env python3
"""Convert eval_agent State JSON files into Q-Evolve trajectory records."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.input_dir).glob("*.json")):
        if p.name == "log.txt":
            continue
        x = json.loads(p.read_text())
        meta = x.get("meta", {})
        conv = x.get("conversations", [])
        if len(conv) < 4:
            continue
        rows.append({
            "id": p.stem,
            "conversations": conv,
            "reward": float(meta.get("reward") or 0.0),
            "finished": bool(meta.get("finished")),
            "success": bool(meta.get("success")),
            "source": "rollout",
        })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
