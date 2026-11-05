#!/usr/bin/env bash
set -euo pipefail
ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-4B
DATA=$ROOT/data/qa/olympiad_rollouts_8/all.jsonl
RUN=$ROOT/runs/olympiad_qevolve
LOG=$RUN/logs
mkdir -p "$LOG"
export PYTHONPATH=$ROOT:$ROOT/QEvolve
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2,3,5
cd "$ROOT"
while [ ! -s "$DATA" ]; do sleep 60; done
$PY - "$DATA" "$RUN" <<'PY'
import json,sys,pathlib
src=pathlib.Path(sys.argv[1]); run=pathlib.Path(sys.argv[2]); rows=[json.loads(x) for x in src.open() if x.strip()]
for name,sel in [('success',lambda r:r.get('meta',{}).get('success') is True),('all',lambda r:True)]:
 out=run/f'{name}.json'; out.parent.mkdir(parents=True,exist_ok=True); json.dump([r for r in rows if sel(r)],out.open('w'),ensure_ascii=False)
print('rollouts',len(rows),'success',sum(r.get('meta',{}).get('success') is True for r in rows))
PY
# 1) successful-trajectory SFT baseline
$PY -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29701 $ROOT/eval_agent/train_sft.py \
 --model_name_or_path "$MODEL" --data_path "$RUN/success.json" --bf16 True --output_dir "$RUN/sft_success" \
 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
 --save_strategy no --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
 --logging_steps 10 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer \
 --tf32 True --model_max_length 4096 --gradient_checkpointing True --lazy_preprocess False 2>&1 | tee "$LOG/sft_success.log"
# 2) critic on all correct and incorrect rollouts
$PY -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29702 $ROOT/QEvolve/train_critic.py \
 --data_path "$DATA" --model_name "$MODEL" --task qa --no_aux_reward --max_length 4096 \
 --max_tokens_per_device 8192 --batch_size 1 --epochs 5 --lr 1e-5 --output_dir "$RUN/critic" \
 --cache_dir "$RUN/cache" --exp_name olympiad --K 4 --gradient_accumulation_steps 1 2>&1 | tee "$LOG/critic.log"
# 3) true critic/GAE advantage; no outcome override
CUDA_VISIBLE_DEVICES=1 $PY -u $ROOT/QEvolve/inf_critic.py --data_path "$DATA" --task qa --no_aux_reward \
 --advantage_base_model "$MODEL" --advantage_model_path "$RUN/critic/iter1/Qwen3-4B" --max_length 4096 \
 --exp_name olympiad --strict 2>&1 | tee "$LOG/advantage.log"
ADV=$ROOT/advantage_inference/qa_traj_tokens/wo_assigned_rewards/olympiad/adv_token_pred-gamma0.95-len4096-olympiad.jsonl
$PY $ROOT/bin/qa/filter_advantage.py --input "$ADV" --output "$RUN/adv_positive.jsonl" | tee "$LOG/filter_advantage.log"
# 4) Q-Evolve policy; all nonzero GAE steps, no qa_outcome_advantage
$PY -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=29703 $ROOT/QEvolve/train_policy.py \
 --data_path "$DATA" --task qa --no_aux_reward --qv_jsonl "$RUN/adv_positive.jsonl" \
 --policy_model "$MODEL" --lr 1e-5 --max_length 4096 --epochs 2 --output_dir "$RUN/policy" \
 --gradient_accumulation_steps 1 --max_tokens_per_device 4096 --max_examples_per_device 2 \
 --dataloader_num_workers 0 --alpha_ul 0 --kl_coef 0.1 --think_token_weight 1.0 \
 --eps_low 0.8 --eps_high 0.4 --neg_mode all --lora 2>&1 | tee "$LOG/policy.log"
echo OLYMPIAD_TRAIN_DONE
