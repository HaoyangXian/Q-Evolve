#!/bin/bash

source env_init.sh

source bin/llama2-7b/alfworld/iter1/config.sh
export WANDB_PROJECT=QEvolve-AdvEstimation       # 设置项目名称


# Standalone advantage estimation: recompute A_t from an existing critic checkpoint
# without retraining (token_estimator.sh already does train+estimate in one go).
# --data_path is the same trajectory jsonl used to train the critic.
$PYTHON QEvolve/inf_critic.py \
    --data_path ${DATA} \
    --task ${TASK} \
    ${AUX_ARG} \
    --advantage_base_model ${BASE_MODEL} \
    --advantage_model_path ${ADV_CKPT:-${CRITIC_CKPT}} \
    --max_length ${MAXLEN} \
    --exp_name ${EXP} \
    --strict

