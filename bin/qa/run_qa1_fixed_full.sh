#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
PYTHON=/data/xhy/miniconda3/envs/qevolve/bin/python
MODEL=/data1/xhy/models/Qwen3-4B
DATA=${ROOT}/data/qa/rollouts_qa1_fixed.jsonl
EXP=qa1_fixed
CRITIC_ROOT=${ROOT}/runs/${EXP}_critic
CRITIC_CKPT=${CRITIC_ROOT}/iter1/Qwen3-4B/checkpoint-final
ADV=${ROOT}/advantage_inference/qa_traj_tokens/wo_assigned_rewards/${EXP}/adv_token_pred-gamma0.95-len4096-${EXP}.jsonl
POLICY_OUT=${ROOT}/runs/${EXP}_policy_4096

export PYTHONPATH=${ROOT}:${ROOT}/QEvolve
export TOKENIZERS_PARALLELISM=false
export INF_MAX_TOKENS=32768
export INF_MAX_EXAMPLES=8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

cd "${ROOT}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=${MASTER_PORT:-29517} \
  "${ROOT}/QEvolve/train_critic.py" \
  --data_path "${DATA}" --model_name "${MODEL}" --task qa --no_aux_reward \
  --max_length 4096 --max_tokens_per_device 8192 --batch_size 1 \
  --epochs 2 --warmup_epochs 0 --lr 1e-5 --output_dir "${CRITIC_ROOT}" \
  --exp_name iter1 --K 4 --gradient_accumulation_steps 1

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u "${ROOT}/QEvolve/inf_critic.py" \
  --data_path "${DATA}" --task qa --no_aux_reward \
  --advantage_base_model "${MODEL}" \
  --advantage_model_path "${CRITIC_ROOT}/iter1/Qwen3-4B" \
  --max_length 4096 --exp_name "${EXP}" --strict

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 --master_port=${POLICY_MASTER_PORT:-29518} \
  "${ROOT}/QEvolve/train_policy.py" \
  --data_path "${DATA}" --task qa --no_aux_reward --qv_jsonl "${ADV}" \
  --policy_model "${MODEL}" --lr 1e-5 --max_length 4096 --epochs 1 \
  --output_dir "${POLICY_OUT}" --gradient_accumulation_steps 1 \
  --max_tokens_per_device 8192 --max_examples_per_device 8 \
  --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 \
  --think_token_weight 1.0 --eps_low 0.8 --eps_high 0.4 \
  --neg_mode drop --qa_outcome_advantage --lora
