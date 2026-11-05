#!/usr/bin/env bash
# 等 Solver-B/C 的 critic 打分都完成后，自动计算三档 AUC + 负对照
set -uo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
log() { echo "$(date +%F_%T) [auc] $*"; }

CRIT_A="$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_A_ms512/adv_token_pred-gamma0.95-len32768-drift_A_ms512.jsonl"
CRIT_B="$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_B_ms64/adv_token_pred-gamma0.95-len32768-drift_B_ms64.jsonl"
CRIT_C="$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_C_ms32/adv_token_pred-gamma0.95-len32768-drift_C_ms32.jsonl"
SOL_A="$ROOT/drift_exp/solver_A_ms512.jsonl"
SOL_B="$ROOT/drift_exp/solver_B_ms64.jsonl"
SOL_C="$ROOT/drift_exp/solver_C_ms32.jsonl"
BASE="$ROOT/data/qa/olympiad_eval200_seed20261031/base.jsonl"

log "等待 critic B/C 打分完成..."
while [ ! -f "$CRIT_B" ] || [ ! -f "$CRIT_C" ]; do
  sleep 60
done
# 额外等 5 秒确保文件写完整
sleep 5
log "critic B/C 已就绪，开始计算三档 AUC + 负对照"

$PY "$ROOT/bin/qa/calc_drift_auc.py" \
  --critic "$CRIT_A" "$CRIT_B" "$CRIT_C" \
  --solvers "$SOL_A" "$SOL_B" "$SOL_C" \
  --base "$BASE" \
  2>&1 | tee "$ROOT/drift_exp/drift_auc_result.txt"

log "AUC 计算完成，结果已写入 drift_exp/drift_auc_result.txt"
