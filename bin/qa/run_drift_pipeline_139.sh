#!/usr/bin/env bash
# 实验A 全自动编排（.139）：
#   Solver-A(512) 立即启动(卡0) → 等校准5档 → 选退化档B/C → Solver-B(卡1)/C(卡2) → critic打分(卡0) → AUC
# 用法（$ROOT 下）: nohup bash bin/qa/run_drift_pipeline_139.sh > drift_exp/pipeline.log 2>&1 &
set -uo pipefail

ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
DRIFT=$ROOT/drift_exp
CALIB_N=30
SOLVER_N=200
BASE=$ROOT/data/qa/olympiad_eval200_seed20261031/base.jsonl

CRITIC_BASE=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/sft_success
CRITIC_HEAD=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B

export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

log() { echo "$(date '+%F %T') [pipeline] $*"; }
gpu_free() { nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
file_lines() { wc -l < "$1" 2>/dev/null || echo 0; }

wait_gpu_free() { # $1=gpu $2=need_mib
  while :; do
    free=$(gpu_free "$1")
    if [ "${free:-0}" -ge "$2" ]; then log "卡$1 空闲 ${free}MiB（需≥$2），放行"; return 0; fi
    sleep 60
  done
}

# ===== 阶段0a：立即启动 Solver-A（512 正常档，不等校准）卡0 =====
if [ "$(file_lines "$DRIFT/solver_A_ms512.jsonl")" -lt "$SOLVER_N" ]; then
  wait_gpu_free 0 20000
  log "启动 Solver-A(ms512) 到卡0"
  GPU=0 nohup bash "$ROOT/bin/qa/run_drift_solver_139.sh" 512 A_ms512 > "$DRIFT/solver_A.launch.log" 2>&1 &
fi

# ===== 阶段0b：等校准 5 档完成 =====
log "等待校准集 5 档完成（每档 $CALIB_N 题）..."
for MS in 512 64 32 16 8; do
  while [ "$(file_lines "$DRIFT/calib_ms${MS}.jsonl")" -lt "$CALIB_N" ]; do sleep 60; done
  log "校准档 ms$MS 完成"
done

# ===== 阶段1：选退化档 B/C =====
log "计算校准 accuracy，选定 A/B/C 档..."
SEL=$($PY - "$DRIFT" <<'PYEOF'
import json, sys, pathlib
sys.path.insert(0, '/data3/xhy/codes/work/paper_repos/Q-Evolve')
from eval_agent.qa_verifier import answers_equal
drift = pathlib.Path(sys.argv[1])
order = [512, 64, 32, 16, 8]
acc = {}
for ms in order:
    f = drift / f"calib_ms{ms}.jsonl"
    if not f.exists():
        acc[ms] = 0.0; continue
    rows = [json.loads(x) for x in f.open() if x.strip()]
    ok = sum(answers_equal(r.get('pred_answer',''), r.get('gold_answer','')) for r in rows)
    acc[ms] = ok/len(rows) if rows else 0.0
print("ACC_TABLE " + " ".join(f"{ms}={acc[ms]:.3f}" for ms in order))
A = 512
B = next((ms for ms in order[1:] if acc[A]-acc[ms] >= 0.15), 32)
C = next((ms for ms in order[1:] if acc[A]-acc[ms] >= 0.30 and acc[ms] < acc[B]), 8)
if B == C:
    C = 8 if B != 8 else 16
print(f"SEL {A} {B} {C}")
PYEOF
)
echo "$SEL" | tee "$DRIFT/calib_selection.txt"
A_MS=$(echo "$SEL" | sed -n 's/^SEL \([0-9]*\) \([0-9]*\) \([0-9]*\)$/\1/p')
B_MS=$(echo "$SEL" | sed -n 's/^SEL \([0-9]*\) \([0-9]*\) \([0-9]*\)$/\2/p')
C_MS=$(echo "$SEL" | sed -n 's/^SEL \([0-9]*\) \([0-9]*\) \([0-9]*\)$/\3/p')
[ -n "$A_MS" ] && [ -n "$B_MS" ] && [ -n "$C_MS" ] || { log "选档解析失败，退出"; exit 4; }
log "选定档位：A=$A_MS  B=$B_MS  C=$C_MS"

# ===== 阶段2：启动 Solver-B(卡1)、Solver-C(卡2) =====
if [ "$(file_lines "$DRIFT/solver_B_ms${B_MS}.jsonl")" -lt "$SOLVER_N" ]; then
  wait_gpu_free 1 20000
  log "启动 Solver-B(ms$B_MS) 到卡1"
  GPU=1 nohup bash "$ROOT/bin/qa/run_drift_solver_139.sh" "$B_MS" "B_ms${B_MS}" > "$DRIFT/solver_B.launch.log" 2>&1 &
fi
if [ "$(file_lines "$DRIFT/solver_C_ms${C_MS}.jsonl")" -lt "$SOLVER_N" ]; then
  wait_gpu_free 2 20000
  log "启动 Solver-C(ms$C_MS) 到卡2"
  GPU=2 nohup bash "$ROOT/bin/qa/run_drift_solver_139.sh" "$C_MS" "C_ms${C_MS}" > "$DRIFT/solver_C.launch.log" 2>&1 &
fi

# ===== 阶段3：等三 Solver 完成 =====
for TAG in A_ms${A_MS} B_ms${B_MS} C_ms${C_MS}; do
  while [ "$(file_lines "$DRIFT/solver_${TAG}.jsonl")" -lt "$SOLVER_N" ]; do sleep 60; done
  log "Solver $TAG 完成（$SOLVER_N 题）"
done

# ===== 阶段4：critic 打分（卡0 串行，需 ~29G）=====
wait_gpu_free 0 30000
ADV_PATHS=()
for TAG in A_ms${A_MS} B_ms${B_MS} C_ms${C_MS}; do
  log "critic 打分 $TAG（卡0）..."
  CUDA_VISIBLE_DEVICES=0 $PY -u "$ROOT/QEvolve/inf_critic_chunked_fast.py" \
    --data_path "$DRIFT/solver_${TAG}.jsonl" \
    --task qa --no_aux_reward \
    --advantage_base_model "$CRITIC_BASE" \
    --advantage_model_path "$CRITIC_HEAD" \
    --max_length 32768 --exp_name "drift_${TAG}" --strict \
    > "$DRIFT/critic_${TAG}.log" 2>&1
  ADV_PATHS+=("$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/drift_${TAG}/adv_token_pred-gamma0.95-len32768-drift_${TAG}.jsonl")
  log "critic $TAG 完成"
done

# ===== 阶段5：AUC + 负对照 =====
log "计算 AUC + 负对照..."
$PY "$ROOT/bin/qa/calc_drift_auc.py" \
  --critic "${ADV_PATHS[0]}" "${ADV_PATHS[1]}" "${ADV_PATHS[2]}" \
  --solvers "$DRIFT/solver_A_ms${A_MS}.jsonl" "$DRIFT/solver_B_ms${B_MS}.jsonl" "$DRIFT/solver_C_ms${C_MS}.jsonl" \
  --base "$BASE" \
  > "$DRIFT/auc_result.txt" 2>&1
log "AUC 计算完成，结果如下："
cat "$DRIFT/auc_result.txt"
log "=== 实验A 全流程结束 ==="
