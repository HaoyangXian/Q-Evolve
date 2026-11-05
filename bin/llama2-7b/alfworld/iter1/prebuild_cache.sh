#!/bin/bash
# One-time: build the trajectory dataset cache (CPU only, no GPU) so all DDP ranks
# load_from_disk instead of each rebuilding. Run before the multi-GPU token_estimator.
source env_init.sh
source bin/llama2-7b/alfworld/iter1/config.sh

$PYTHON QEvolve/train_critic.py \
    --data_path ${DATA} \
    --model_name ${BASE_MODEL} \
    --task ${TASK} \
    ${AUX_ARG} \
    --max_length ${MAXLEN} \
    --exp_name ${EXP} \
    --output_dir ${CRITIC_DIR} \
    --cache_dir ${CACHE_DIR} \
    --preprocess_only
echo "PREBUILD_DONE rc=$?"
