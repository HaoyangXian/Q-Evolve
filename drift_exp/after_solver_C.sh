#!/usr/bin/env bash
# Solver-C(ms32, pid 671554) 完成后：critic 打分(卡3) → 占卡3
set -uo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

TARGET_PID=${1:-671554}
CRITIC_BASE=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/sft_success
CRITIC_HEAD=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B

log() { echo "$(date +%F_%T) [after-C] $*"; }

log "等待 Solver-C pid=$TARGET_PID 完成..."
while kill -0 "$TARGET_PID" 2>/dev/null; do sleep 60; done
log "Solver-C 已完成，开始 critic 打分（卡3）"

CUDA_VISIBLE_DEVICES=3 $PY -u $ROOT/QEvolve/inf_critic_chunked_fast.py \
  --data_path $ROOT/drift_exp/solver_C_ms32.jsonl \
  --task qa --no_aux_reward \
  --advantage_base_model "$CRITIC_BASE" \
  --advantage_model_path "$CRITIC_HEAD" \
  --max_length 32768 --exp_name drift_C_ms32 --strict \
  > $ROOT/drift_exp/critic_C_ms32.log 2>&1
log "critic C_ms32 打分完成"

# 占卡3
nohup "$PY" /data3/xhy/codes/gpu_reserve.py --gpus 3 --fraction 0.85 \
  > /data3/xhy/codes/gpu_reserve_gpu3.log 2>&1 < /dev/null &
log "已占住卡3"
