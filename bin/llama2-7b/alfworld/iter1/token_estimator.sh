#!/bin/bash

source env_init.sh
source bin/llama2-7b/alfworld/iter1/config.sh
export WANDB_PROJECT=QEvolve-AdvEstimation
export WANDB_MODE=offline

# GPU visibility diagnostic (helps tell a bad node from a torch/CUDA_VISIBLE_DEVICES issue under DDP)
echo "[gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"; nvidia-smi -L 2>&1 | head -8

# 1) Evolve Critic: learn implicit Q/V (in-memory preprocessing; pass --cache_dir to opt into a disk cache)
$PYTHON -m torch.distributed.run --nproc_per_node=${NPROC:-4} --master_port=${MASTER_PORT:-29500} QEvolve/train_critic.py \
    --data_path ${DATA} \
    --model_name ${BASE_MODEL} \
    --lr 1e-5 \
    --max_length ${MAXLEN} \
    --max_tokens_per_device ${MAXTOK:-65536} \
    --batch_size ${BS:-1} \
    --epochs ${EPOCHS:-20} \
    --output_dir ${CRITIC_DIR} \
    --exp_name ${EXP} \
    --gradient_accumulation_steps ${GA:-1} \
    --task ${TASK} \
    ${AUX_ARG} \
    --wandb

# 2) Estimate process rewards A_t via GAE (fused; single GPU; same jsonl + the critic just trained)
#    -> advantage_inference/${TASK}_traj_tokens/w_assigned_rewards/${EXP}/adv_token_pred-gamma0.95-len${MAXLEN}-${EXP}.jsonl
$PYTHON QEvolve/inf_critic.py \
    --data_path ${DATA} \
    --task ${TASK} \
    ${AUX_ARG} \
    --advantage_base_model ${BASE_MODEL} \
    --advantage_model_path ${CRITIC_CKPT} \
    --max_length ${MAXLEN} \
    --exp_name ${EXP} \
    --strict
