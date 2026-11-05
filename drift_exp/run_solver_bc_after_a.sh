#!/usr/bin/env bash
# 等卡0(Solver-A)释放后启动 Solver-B(ms64)。
# 注意：Solver-C(ms32) 已改由卡3 独立启动（after_solver_C.sh 编排），此处不再重复启动。
set -uo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
DRIFT=$ROOT/drift_exp
SOLVER_N=200
log() { echo "$(date +%F_%T) [bc] $*"; }
gpu_free() { nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d " " ; }
file_lines() { wc -l < "$1" 2>/dev/null || echo 0; }

wait_gpu_free() {
  while :; do
    free=$(gpu_free "$1")
    if [ "${free:-0}" -ge "$2" ]; then log "卡$1 空闲 ${free}MiB（需≥$2），放行"; return 0; fi
    sleep 60
  done
}

# 等 Solver-A 完成(满200)后，卡0 释放
log "等待 Solver-A 完成(当前 $(file_lines "$DRIFT/solver_A_ms512.jsonl")/200)..."
while [ "$(file_lines "$DRIFT/solver_A_ms512.jsonl")" -lt "$SOLVER_N" ]; do sleep 60; done
log "Solver-A 已完成，等待卡0 释放显存..."

# Solver-B(ms64) -> 卡0
if [ "$(file_lines "$DRIFT/solver_B_ms64.jsonl")" -lt "$SOLVER_N" ]; then
  wait_gpu_free 0 20000
  log "启动 Solver-B(ms64) 到卡0"
  GPU=0 WALL_CLOCK=2400 OFFSET=0 LIMIT=200 nohup bash "$ROOT/bin/qa/run_drift_solver_139.sh" 64 B_ms64 > "$DRIFT/solver_B.launch.log" 2>&1 &
fi

# 等 Solver-B 完成
log "等待 Solver-B 完成..."
while [ "$(file_lines "$DRIFT/solver_B_ms64.jsonl")" -lt "$SOLVER_N" ]; do sleep 60; done
log "Solver-B 已完成（Solver-C 已由卡3 独立运行，此处不重复启动）"

# Solver-B 完成后：critic 打分(卡0) → 占卡0
CRITIC_BASE=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/sft_success
CRITIC_HEAD=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python

log "Solver-B 完成后 critic 打分（卡0）..."
CUDA_VISIBLE_DEVICES=0 $PY -u "$ROOT/QEvolve/inf_critic_chunked_fast.py" \
  --data_path "$DRIFT/solver_B_ms64.jsonl" \
  --task qa --no_aux_reward \
  --advantage_base_model "$CRITIC_BASE" \
  --advantage_model_path "$CRITIC_HEAD" \
  --max_length 32768 --exp_name drift_B_ms64 --strict \
  > "$DRIFT/critic_B_ms64.log" 2>&1
log "critic B_ms64 打分完成"

# 占卡0
nohup "$PY" /data3/xhy/codes/gpu_reserve.py --gpus 0 --fraction 0.85 \
  > /data3/xhy/codes/gpu_reserve_gpu0.log 2>&1 < /dev/null &
log "已占住卡0"
