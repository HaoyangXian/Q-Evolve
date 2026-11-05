#!/usr/bin/env bash
# Branching Continuation Pilot 启动脚本（2 卡并行）
set -uo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
cd "$ROOT"
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
export PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false

OUT=$ROOT/drift_exp/branch_pilot
mkdir -p "$OUT"
rm -f "$OUT"/*.jsonl "$OUT"/*.log

for SHARD in 0 1; do
  if [ "$SHARD" = "0" ]; then GPU=0; else GPU=3; fi
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY -u drift_exp/branch_pilot.py \
    --mode all \
    --shard $SHARD --nshards 2 \
    --n_questions 20 \
    --n_traj 3 \
    --max_steps 96 \
    --step_tokens 1024 \
    --temperature 0.8 \
    --prefixes 16 32 \
    --cont_budget 8 \
    --n_cont 4 \
    --outdir "$OUT" \
    > "$OUT/shard${SHARD}.log" 2>&1 < /dev/null &
  echo "shard $SHARD -> GPU $GPU, PID=$!"
  sleep 3
done

echo "已启动，日志目录: $OUT"
