#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
PYTHON=/data/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data/xhy/codes/models/Llama-2-7b-chat-hf
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PARTS=${PARTS:-8}
RUN_ROOT=${RUN_ROOT:-${ROOT}/runs/paper_repro}
LOG_ROOT=${RUN_ROOT}/logs
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

export PYTHONPATH="$ROOT:$ROOT/QEvolve:$ROOT/envs/webshop/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export ALFWORLD_DATA="$ROOT/eval_agent/data/alfworld"
export WEBSHOP_DATA="$ROOT/env_data/webshop"
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PYTHONPATH="$ROOT:$ROOT/QEvolve:$ROOT/envs/webshop/src:/data/xhy/codes/cache/pyserini_site:/data/xhy/codes/cache/pyjnius_site:/data/xhy/codes/cache/onnxruntime_site:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_PORT=${MASTER_PORT:-29601}
export WANDB_MODE=offline

prepare_manifests() {
  "$PYTHON" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
wd=root/'eval_agent/data/webshop'; wd.mkdir(parents=True,exist_ok=True)
json.dump(list(range(1824)),(wd/'train_indices.json').open('w'))
json.dump(list(range(1824,2024)),(wd/'test_indices.json').open('w'))
json.dump(list(range(1824,2024)),(wd/'test_indices_500.json').open('w'))
sw=root/'eval_agent/data/sciworld'; sw.mkdir(parents=True,exist_ok=True)
tasks=json.load(open('/data/xhy/miniconda3/envs/pettingllms/lib/python3.12/site-packages/scienceworld/tasks.json'))
mapping={int(x['task_id'].split('-')[0]):x['task_name'] for x in tasks}
rows=json.load(open(root/'data/train/sciworld/sciworld_sft.json'))
ids=[]
for r in rows:
    a,b=str(r['id']).split('_',1)
    if int(a) in mapping: ids.append([mapping[int(a)],int(b)])
ids=list(dict.fromkeys(ids))
json.dump(ids[:1200],(sw/'train_indices.json').open('w'))
json.dump(ids[1200:1341],(sw/'dev_indices.json').open('w'))
json.dump(ids[1341:],(sw/'test_indices.json').open('w'))
json.dump({v:k for k,v in mapping.items()},(sw/'taskname2id.json').open('w'))
json.dump({'webshop_train':1824,'webshop_test':200,'sciworld_train':len(ids[:1200]),'sciworld_dev':len(ids[1200:1341]),'sciworld_test':len(ids[1341:])},(root/'data/paper_repro_split_manifest.json').open('w'),indent=2)
PY
}

start_server() {
  local model_path=$1
  local log=$2
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m sglang.launch_server --model-path "$model_path" --port 21001 >"$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if curl -fsS http://127.0.0.1:21001/get_model_info >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 80 "$log"; return 1; fi
    sleep 5
  done
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

collect_rollouts() {
  local task=$1
  local sft=$2
  local out_dir="outputs4/Llama-2-7b-chat-hf/${task}paper_${task}_train1shot"
  rm -f "$LOG_ROOT/${task}_explore.log"
  start_server "$sft" "$LOG_ROOT/${task}_sglang_explore.log"
  trap stop_server EXIT
  local pids=()
  for part in $(seq 0 $((PARTS-1))); do
    "$PYTHON" -m eval_agent.inference \
      --agent_config sglang_explore0 --agent_path eval_agent/configs/model \
      --exp_name "paper_${task}_train" --exp_path eval_agent/configs/task \
      --exp_config "$task" --split train --part_num "$PARTS" --part_idx "$part" \
      --model_name Llama-2-7b-chat-hf --num_icl_examples 1 --force_first --override \
      >>"$LOG_ROOT/${task}_explore.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p"; done
  stop_server
  "$PYTHON" bin/convert_states_to_traj.py --input_dir "$out_dir" --output "$RUN_ROOT/${task}_rollouts.json"
  "$PYTHON" bin/merge_paper_traj.py \
    --expert "data/train/${task}/${task}_sft.json" \
    --rollout "$RUN_ROOT/${task}_rollouts.json" \
    --output "$RUN_ROOT/${task}_hybrid.json"
}

train_task() {
  local task=$1
  local maxlen=$2
  local port=$3
  local sft="$RUN_ROOT/${task}_sft"
  local critic="$RUN_ROOT/${task}_critic"
  local adv="$ROOT/advantage_inference/${task}_traj_tokens/w_assigned_rewards/paper_${task}/adv_token_pred-gamma0.95-len${maxlen}-paper_${task}.jsonl"
  local policy="$RUN_ROOT/${task}_policy"
  local merged="$RUN_ROOT/${task}_policy_merged"
  local data="$RUN_ROOT/${task}_hybrid.json"

  if [[ ! -f "$sft/config.json" ]]; then
    local acc=$((64 / NPROC / 2))
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --master_port="$port" eval_agent/train_sft.py \
      --model_name_or_path "$MODEL" --data_path "data/train/${task}/${task}_sft.json" \
      --bf16 True --output_dir "$sft" --num_train_epochs 3 \
      --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps "$acc" --save_strategy no --learning_rate 2e-5 \
      --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 10 \
      --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
      --tf32 True --model_max_length "$maxlen" --gradient_checkpointing True --lazy_preprocess False \
      2>&1 | tee "$LOG_ROOT/${task}_sft.log"
  fi
  [[ -f "$data" ]] || collect_rollouts "$task" "$sft"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --master_port="$((port+1))" QEvolve/train_critic.py \
    --data_path "$data" --model_name "$sft" --task "$task" --use_aux_reward \
    --max_length "$maxlen" --max_tokens_per_device 8192 --batch_size 1 --epochs 20 --lr 1e-5 \
    --output_dir "$critic" --exp_name paper_${task} --gradient_accumulation_steps 1 \
    2>&1 | tee "$LOG_ROOT/${task}_critic.log"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" QEvolve/inf_critic.py --data_path "$data" --task "$task" --use_aux_reward \
    --advantage_base_model "$sft" --advantage_model_path "$critic/iter1/$(basename "$sft")" \
    --max_length "$maxlen" --exp_name paper_${task} --strict 2>&1 | tee "$LOG_ROOT/${task}_advantage.log"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --master_port="$((port+2))" QEvolve/train_policy.py \
    --data_path "$data" --task "$task" --use_aux_reward --qv_jsonl "$adv" \
    --policy_model "$sft" --lr 1e-5 --max_length "$maxlen" --epochs 3 --output_dir "$policy" \
    --gradient_accumulation_steps 1 --max_tokens_per_device 4096 --max_examples_per_device 2 \
    --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 --think_token_weight 0.7 \
    --eps_low 0.8 --eps_high 0.4 --lora 2>&1 | tee "$LOG_ROOT/${task}_policy.log"
  "$PYTHON" bin/merge_lora.py --base "$sft" --adapter "$policy" --output "$merged" 2>&1 | tee "$LOG_ROOT/${task}_merge.log"
}

evaluate_task() {
  local task=$1
  local model=$2
  local tag=$3
  start_server "$model" "$LOG_ROOT/${task}_${tag}_sglang.log"
  trap stop_server EXIT
  "$PYTHON" -m eval_agent.inference --agent_config sglang_sft_long --agent_path eval_agent/configs/model \
    --exp_name "paper_${task}_${tag}" --exp_path eval_agent/configs/task --exp_config "$task" \
    --split dev --model_name Llama-2-7b-chat-hf --num_icl_examples 1 --force_first --override \
    2>&1 | tee "$LOG_ROOT/${task}_${tag}_dev.log"
  "$PYTHON" -m eval_agent.inference --agent_config sglang_sft_long --agent_path eval_agent/configs/model \
    --exp_name "paper_${task}_${tag}" --exp_path eval_agent/configs/task --exp_config "$task" \
    --split test --model_name Llama-2-7b-chat-hf --num_icl_examples 1 --force_first --override \
    2>&1 | tee "$LOG_ROOT/${task}_${tag}_test.log"
  stop_server
}

prepare_manifests
for spec in "alfworld 3600 29610" "webshop 4096 29620" "sciworld 4096 29630"; do
  read -r task maxlen port <<<"$spec"
  train_task "$task" "$maxlen" "$port"
  evaluate_task "$task" "$RUN_ROOT/${task}_sft" sft
  evaluate_task "$task" "$RUN_ROOT/${task}_policy_merged" policy
done
echo "PAPER_REPRO_DONE $(date '+%F %T')" | tee "$RUN_ROOT/DONE"
