#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
PYTHON=/data/xhy/miniconda3/envs/qevolve/bin/python
MODEL=/data1/xhy/models/Qwen3-4B
DATA=${ROOT}/data/qa/rollouts_qa1_fixed.jsonl
ADV=${ROOT}/advantage_inference/qa_traj_tokens/wo_assigned_rewards/qa1_fixed/adv_token_pred-gamma0.95-len4096-qa1_fixed.jsonl
OUT=${ROOT}/runs/qa1_policy_4096_fixed

export PYTHONPATH=${ROOT}:${ROOT}/QEvolve
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

exec ${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=${POLICY_MASTER_PORT:-29518} \
  ${ROOT}/QEvolve/train_policy.py \
  --data_path ${DATA} --task qa --no_aux_reward --qv_jsonl ${ADV} \
  --policy_model ${MODEL} --lr 1e-5 --max_length 4096 --epochs 1 \
  --output_dir ${OUT} --gradient_accumulation_steps 1 \
  --max_tokens_per_device 8192 --max_examples_per_device 8 \
  --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 \
  --think_token_weight 1.0 --eps_low 0.8 --eps_high 0.4 \
  --neg_mode drop --qa_outcome_advantage --lora
