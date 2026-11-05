#!/bin/bash

source env_init.sh
export WANDB_PROJECT=QEvolve

mkdir -p logs

task=alfworld # webshop, alfworld, alfworld
exp_name=qlass

node_num=1  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=$PRJS/${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sg_worker_port=21001
sft_model_name=${PRJS}/QLASS/meta-llama/Llama-2-7b-chat-hf/sft-Llama-2-7b-chat-hf-alfworld-sft/
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${sft_model_name} --port ${sg_worker_port} >> logs/${exp_name}-sg_worker.log 2>&1 &

sg_worker_pid=$!
cleanup() {
  echo "[cleanup] stopping sglang server pid=${sg_worker_pid}"
  if kill -0 "${sg_worker_pid}" 2>/dev/null; then
    kill "${sg_worker_pid}" 2>/dev/null || true
    for i in $(seq 1 20); do
      kill -0 "${sg_worker_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${sg_worker_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
for i in $(seq 1 60); do
  (echo > /dev/tcp/127.0.0.1/${sg_worker_port}) >/dev/null 2>&1 && break
  sleep 1
done

python -m eval_agent.inference --agent_config sglang_explore0 \
    --model_name Llama-2-7b-chat-alfworld-sft --exp_config alfworld \
    --split train --num_icl_examples 1 --exp_name qlass_alfworld_explore_by_sft-seed-0 --force_first &
python -m eval_agent.inference --agent_config sglang_explore0 \
    --model_name Llama-2-7b-chat-alfworld-sft --exp_config alfworld \
    --split train --num_icl_examples 1 --exp_name qlass_alfworld_explore_by_sft-seed-1 --force_first &
python -m eval_agent.inference --agent_config sglang_explore0 \
    --model_name Llama-2-7b-chat-alfworld-sft --exp_config alfworld \
    --split train --num_icl_examples 1 --exp_name qlass_alfworld_explore_by_sft-seed-2 --force_first &

wait
