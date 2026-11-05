#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
INPUT=$ROOT/data/qa/olympiadbench.jsonl
OUT=$ROOT/data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_small
mkdir -p "$OUT"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
pids=(); off=0
for part in 0 1 2 3 4 5 6 7; do
  lim=4
  CUDA_VISIBLE_DEVICES=$part $PY -u $ROOT/eval_agent/qa_rollout.py --input "$INPUT" --output "$OUT/part${part}.jsonl" --model "$MODEL" --offset "$off" --limit "$lim" --samples 8 --max_new_tokens 8192 --step_max_new_tokens 1024 --max_steps 256 --temperature 0.8 --termination_mode strict >"$OUT/part${part}.log" 2>&1 & pids+=("$!"); off=$((off+lim))
done
for p in "${pids[@]}"; do wait "$p"; done
$PY - <<'PY'
from pathlib import Path
p=Path('/data3/xhy/codes/work/paper_repos/Q-Evolve/data/qa/olympiad_rollouts_qwen3_8b_temp08_iterative1024_small'); out=p/'all.jsonl'
with out.open('w') as f:
 for i in range(8): f.write((p/f'part{i}.jsonl').read_text())
print('merged',sum(1 for _ in out.open()),out)
PY
