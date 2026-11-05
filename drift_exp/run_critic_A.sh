#!/usr/bin/env bash
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
CUDA_VISIBLE_DEVICES=3 nohup $PY -u $ROOT/QEvolve/inf_critic_chunked_fast.py \
  --data_path $ROOT/drift_exp/solver_A_ms512.jsonl \
  --task qa --no_aux_reward \
  --advantage_base_model $ROOT/runs/olympiad_qwen3_8b_iterative1024_small/sft_success \
  --advantage_model_path $ROOT/runs/olympiad_qwen3_8b_iterative1024_small/critic/olympiad/Qwen3-8B \
  --max_length 32768 --exp_name drift_A_ms512 --strict \
  > $ROOT/drift_exp/critic_A_ms512.log 2>&1 < /dev/null &
echo "critic PID=$!"
