#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
PYTHON=/data/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data/xhy/codes/models/Qwen3-4B
DATA=${ROOT}/data/qa/rollouts_qa1_fixed.jsonl
EXP=qa1_formal8
NPROC=8
GPUS=0,1,2,3,4,5,6,7
CRITIC_ROOT=${ROOT}/runs/${EXP}_critic
ADV=${ROOT}/advantage_inference/qa_traj_tokens/wo_assigned_rewards/${EXP}/adv_token_pred-gamma0.95-len4096-${EXP}.jsonl
POLICY_OUT=${ROOT}/runs/${EXP}_policy_8192
LOG_ROOT=${ROOT}/runs/${EXP}_logs
CACHE_ROOT=${ROOT}/cache/${EXP}

export PYTHONPATH=${ROOT}:${ROOT}/QEvolve
export TOKENIZERS_PARALLELISM=false
export INF_MAX_TOKENS=32768
export INF_MAX_EXAMPLES=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPUS}
mkdir -p "${LOG_ROOT}"
cd "${ROOT}"

echo "[$(date '+%F %T')] critic start" | tee "${LOG_ROOT}/status.log"
${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${NPROC} --master_port=29641 \
  "${ROOT}/QEvolve/train_critic.py" \
  --data_path "${DATA}" --model_name "${MODEL}" --task qa --no_aux_reward \
  --max_length 4096 --max_tokens_per_device 8192 --batch_size 1 \
  --epochs 5 --warmup_epochs 0 --lr 1e-5 \
  --output_dir "${CRITIC_ROOT}" --cache_dir "${CACHE_ROOT}" --exp_name iter1 --K 4 \
  --gradient_accumulation_steps 1 2>&1 | tee "${LOG_ROOT}/critic.log"
echo "[$(date '+%F %T')] critic done" | tee -a "${LOG_ROOT}/status.log"

echo "[$(date '+%F %T')] advantage start" | tee -a "${LOG_ROOT}/status.log"
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u "${ROOT}/QEvolve/inf_critic.py" \
  --data_path "${DATA}" --task qa --no_aux_reward \
  --advantage_base_model "${MODEL}" \
  --advantage_model_path "${CRITIC_ROOT}/iter1/Qwen3-4B" \
  --max_length 4096 --exp_name "${EXP}" --strict 2>&1 | tee "${LOG_ROOT}/advantage.log"
echo "[$(date '+%F %T')] advantage done" | tee -a "${LOG_ROOT}/status.log"

echo "[$(date '+%F %T')] policy start" | tee -a "${LOG_ROOT}/status.log"
${PYTHON} -m torch.distributed.run --standalone --nproc_per_node=${NPROC} --master_port=29642 \
  "${ROOT}/QEvolve/train_policy.py" \
  --data_path "${DATA}" --task qa --no_aux_reward --qv_jsonl "${ADV}" \
  --policy_model "${MODEL}" --lr 1e-5 --max_length 4096 --epochs 2 \
  --output_dir "${POLICY_OUT}" --gradient_accumulation_steps 1 \
  --max_tokens_per_device 8192 --max_examples_per_device 8 \
  --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 \
  --think_token_weight 1.0 --eps_low 0.8 --eps_high 0.4 \
  --neg_mode drop --qa_outcome_advantage --lora 2>&1 | tee "${LOG_ROOT}/policy.log"
echo "[$(date '+%F %T')] policy done" | tee -a "${LOG_ROOT}/status.log"
