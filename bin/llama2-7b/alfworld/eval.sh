#!/bin/bash

source env_init.sh
export WANDB_PROJECT=QEvolve

# The sglang server and the inference client talk over localhost. On clusters
# that route outbound HTTP through a proxy (http_proxy/https_proxy), requests to
# 127.0.0.1 would otherwise be sent to the proxy and rejected, so exempt local
# traffic explicitly.
export no_proxy="127.0.0.1,localhost,${no_proxy}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY}"

mkdir -p logs

# Base model used for SFT (alternatives: meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-3.2-3B-Instruct)
MODEL_PATH=eval_agent/qlass-Llama-2-7b-chat-hf-alfworld-sft
model_name=qlass-7b-alfworld-sft

task=alfworld   # webshop, scienceworld, alfworld
exp_name=qlass
node_num=1      # number of GPUs
sg_worker_port=21001

# Policy checkpoint to evaluate: ${save_dir}${sft_model_name}
save_dir=${SCRATCH_ROOT}/QLASS-Adv/policy-alfworld/
sft_model_name=${EVAL_CKPT:-v39-h18-noauxpolicy-weightfix-advfilterneg0.5-assigned_reward-weighted-lambda_think0.7-advclip0.8low-0.4high-kl0.1-nolambdaingae-frozenref/}

# Launch the sglang server for the policy checkpoint
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs/${exp_name}-sg_worker.log 2>&1 &
sg_worker_pid=$!

cleanup() {
  echo "[cleanup] stopping sglang server pid=${sg_worker_pid}"
  if kill -0 "${sg_worker_pid}" 2>/dev/null; then
    kill "${sg_worker_pid}" 2>/dev/null || true
    for ((i=1; i<=20; i++)); do
      kill -0 "${sg_worker_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -9 "${sg_worker_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# wait for the server to come up (poll instead of a fixed sleep; a 13G model can
# take well over a minute to load on a cold cache)
echo "[wait] polling sglang server on port ${sg_worker_port} ..."
for ((i=1; i<=120; i++)); do
  if curl -fsS "http://localhost:${sg_worker_port}/health_generate" >/dev/null 2>&1 \
     || curl -fsS "http://localhost:${sg_worker_port}/get_model_info" >/dev/null 2>&1; then
    echo "[wait] server is up after ~$((i*5))s"
    break
  fi
  if ! kill -0 "${sg_worker_pid}" 2>/dev/null; then
    echo "[wait] sglang server process died; see logs/${exp_name}-sg_worker.log" >&2
    tail -n 50 "logs/${exp_name}-sg_worker.log" >&2 || true
    exit 1
  fi
  sleep 5
done

# Evaluate on test and dev splits in parallel
python -m eval_agent.inference --agent_config sglang_sft_long --model_name Llama-2-7b-chat-${task}-${sft_model_name}-test --exp_config alfworld --split test --num_icl_examples 1 --exp_name qlass_eval_sft --force_first --override &
infer1_pid=$!
python -m eval_agent.inference --agent_config sglang_sft_long --model_name Llama-2-7b-chat-${task}-${sft_model_name}-dev  --exp_config alfworld --split dev  --num_icl_examples 1 --exp_name qlass_eval_sft --force_first --override &
infer2_pid=$!

# wait only for the two inference jobs, then shut the server down
wait "${infer1_pid}" "${infer2_pid}"
cleanup
exit 0
