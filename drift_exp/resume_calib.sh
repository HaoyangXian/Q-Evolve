#!/usr/bin/env bash
set -uo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
BASE=/data3/xhy/codes/models/Qwen3-8B
INPUT=$ROOT/data/qa/olympiadbench_eval200_seed20261031.jsonl
OUTDIR=$ROOT/drift_exp
GPU=2
WALL_CLOCK=2400

# 档位定义：ms512 补 10 题(offset20)，其余四档全新 30 题(offset0)
run_one() {
  local MS=$1 OFFSET=$2 LIMIT=$3 APPEND=$4
  local OUT=$OUTDIR/calib_ms${MS}.jsonl
  local APP=""
  [ "$APPEND" = "1" ] && APP="--append"
  echo "$(date '+%F %T') [calib] 档 ms$MS offset=$OFFSET limit=$LIMIT append=$APPEND"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$ROOT"/eval_agent/qa_rollout_chunked_context.py \
    --input "$INPUT" --output "$OUT" --model "$BASE" \
    --offset "$OFFSET" --limit "$LIMIT" \
    --samples 1 --temperature 0 --termination_mode strict \
    --wall_clock_timeout "$WALL_CLOCK" $APP \
    --max_steps "$MS" --step_max_new_tokens 1024 --max_new_tokens 8192 --context_max_tokens 32768 \
    2>&1 | tee "$OUTDIR/calib_ms${MS}.log"
  echo "$(date '+%F %T') [calib] 档 ms$MS 完成，共 $(wc -l < "$OUT") 条"
}

# 1) ms512 补 10 题（第21-30题）
run_one 512 20 10 1
# 2) 其余四档全新
run_one 64 0 30 0
run_one 32 0 30 0
run_one 16 0 30 0
run_one 8 0 30 0
echo "[calib resume done] 全校准完成"
