#!/usr/bin/env bash
# 实验A 校准集：30 题 × 5 档预算，测 raw accuracy 以选定退化档位
# 用法: GPU=2 bash run_drift_calib_139.sh
set -euo pipefail

ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
BASE=/data3/xhy/codes/models/Qwen3-8B
INPUT=$ROOT/data/qa/olympiadbench_eval200_seed20261031.jsonl

GPU=${GPU:-2}
WALL_CLOCK=${WALL_CLOCK:-2400}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-30}
OUTDIR=$ROOT/drift_exp
mkdir -p "$OUTDIR"

for MS in 512 64 32 16 8; do
  echo "$(date '+%F %T') [calib] 开始 max_steps=$MS"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$ROOT"/eval_agent/qa_rollout_chunked_context.py \
    --input "$INPUT" \
    --output "$OUTDIR/calib_ms${MS}.jsonl" \
    --model "$BASE" \
    --offset "$OFFSET" --limit "$LIMIT" \
    --samples 1 --temperature 0 --termination_mode strict \
    --append --wall_clock_timeout "$WALL_CLOCK" \
    --max_steps "$MS" --step_max_new_tokens 1024 --max_new_tokens 8192 --context_max_tokens 32768 \
    2>&1 | tee "$OUTDIR/calib_ms${MS}.log"
  echo "$(date '+%F %T') [calib] 完成 max_steps=$MS"
done
echo "[calib done] 校准集全部完成，输出 drift_exp/calib_ms{512,64,32,16,8}.jsonl"
