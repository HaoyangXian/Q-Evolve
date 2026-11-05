#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
SMALL="$ROOT/runs/olympiad_qwen3_8b_iterative1024_small"
FULL="$ROOT/runs/olympiad_qwen3_8b_iterative1024_full"
LOG="$FULL/auto_start.log"
mkdir -p "$FULL"
while [ ! -f "$SMALL/DONE" ]; do sleep 60; done
printf 'small_done %s\n' "$(date)" >> "$LOG"
if [ ! -f "$SMALL/EVAL_DONE" ]; then
  bash "$ROOT/bin/qa/run_olympiad_eval_small_iterative1024.sh" >> "$SMALL/eval.log" 2>&1
  touch "$SMALL/EVAL_DONE"
  printf 'small_eval_done %s\n' "$(date)" >> "$LOG"
fi
if [ ! -f "$FULL/STARTED" ]; then
  touch "$FULL/STARTED"
  cd "$ROOT"
  nohup bash bin/qa/run_olympiad_rollouts_qwen3_8b_full_iterative1024.sh > "$FULL/rollout_launcher.log" 2>&1 & echo "full_rollout_launcher=$!" >> "$LOG"
  nohup bash bin/qa/run_olympiad_train_qwen3_8b_full_iterative1024.sh > "$FULL/train_launcher.log" 2>&1 & echo "full_train_launcher=$!" >> "$LOG"
  printf 'full_started %s\n' "$(date)" >> "$LOG"
fi
