#!/bin/bash

source env_init.sh

source bin/llama2-7b/alfworld/iter1/config.sh
export WANDB_PROJECT=QEvolve-Policy-Alfworld
export WANDB_MODE=offline
export NPROC=${NPROC:-2}

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# MASTER_PORT is exported by env_init.sh (no seq/ss dependency)

# main results
$PYTHON -m torch.distributed.run  --nproc_per_node=${NPROC:-1} --master_port=$MASTER_PORT QEvolve/train_policy.py \
    --data_path ${DATA} \
    --task ${TASK} \
    ${AUX_ARG} \
    --qv_jsonl ${ADV} \
    --policy_model eval_agent/qlass-Llama-2-7b-chat-hf-alfworld-sft \
    --lr 1e-5 \
    --max_length 3600 \
    --epochs ${EPOCHS_POLICY:-3} \
    --output_dir ${SCRATCH_ROOT}/QLASS-Adv/policy-alfworld/v39-h18-noauxpolicy-weightfix-advfilterneg0.5-assigned_reward-weighted-lambda_think0.7-advclip0.8low-0.4high-kl0.1-nolambdaingae-frozenref/ \
    --gradient_accumulation_steps 1 \
    --max_tokens_per_device ${PMAXTOK:-7200} \
    --max_examples_per_device ${PMAXEX:-2} \
    --alpha_ul 0 \
    --kl_coef 0.1 \
    --think_token_weight 0.7 \
    --eps_low 0.8 \
    --eps_high 0.4 \
    --wandb
