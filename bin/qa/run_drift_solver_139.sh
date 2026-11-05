#!/usr/bin/env bash
# 实验A：跑一个退化档位的 Solver（参数化预算）
# 用法: bash run_drift_solver_139.sh <max_steps> [tag]
#   bash run_drift_solver_139.sh 512 A_ms512
# 产出: drift_exp/solver_<tag>.jsonl
set -euo pipefail

ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
BASE=/data3/xhy/codes/models/Qwen3-8B
INPUT=$ROOT/data/qa/olympiadbench_eval200_seed20261031.jsonl

GPU=${GPU:-2}
WALL_CLOCK=${WALL_CLOCK:-2400}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-200}
MS=${1:?usage: max_steps}
TAG=${2:-ms${MS}}

OUTDIR=$ROOT/drift_exp
mkdir -p "$OUTDIR"
OUT=$OUTDIR/solver_${TAG}.jsonl

# 冻结协议：temperature 0 / strict / samples 1；仅 max_steps 可变
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$ROOT"/eval_agent/qa_rollout_chunked_context.py \
  --input "$INPUT" \
  --output "$OUT" \
  --model "$BASE" \
  --offset "$OFFSET" --limit "$LIMIT" \
  --samples 1 --temperature 0 --termination_mode strict \
  --wall_clock_timeout "$WALL_CLOCK" \
  --append \
  --max_steps "$MS" --step_max_new_tokens 1024 --max_new_tokens 8192 --context_max_tokens 32768 \
  2>&1 | tee "$OUTDIR/solver_${TAG}.log"

echo "[done] $OUT"
