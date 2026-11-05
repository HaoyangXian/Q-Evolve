# Shared config for this iter's pipeline: Evolve-Critic -> estimate A_t -> Evolve-Policy.
# Sourced by token_estimator / infer_qv / policy_extraction (run from repo root).
TASK=alfworld
BASE_MODEL=eval_agent/qlass-Llama-2-7b-chat-hf-alfworld-sft
# iter1 merged data (3x self-explore parts by the SFT model + direct-mapped SFT demos, 13082 trajs).
# NOTE: the original merged_alfworld_q-full-h18.jsonl was produced on the previous cluster and
# no longer exists anywhere; this is the January llama2-7b iter1 dataset it derived from.
DATA=data/train/alfworld/merged_alfworld_q-iter1.jsonl
EXP=iter1
MAXLEN=3600
GAMMA=0.95
CRITIC_DIR=${SCRATCH_ROOT}/QLASS-Adv/critic-alfworld/
CRITIC_CKPT=${CRITIC_DIR%/}/${EXP}/${BASE_MODEL##*/}
# Pre-built trajectory dataset cache (build once with --preprocess_only; all DDP
# ranks load_from_disk instead of each rebuilding). Resolves to
# ${CACHE_DIR}/w_assigned_rewards/${EXP} inside train_critic.
CACHE_DIR=${SCRATCH_ROOT}/traj_cache
USE_AUX=1   # 1 = apply retrospective auxiliary reward (in memory); 0 = env reward only
if [ "$USE_AUX" = "1" ]; then SUB=w_assigned_rewards; AUX_ARG="--use_aux_reward"; else SUB=wo_assigned_rewards; AUX_ARG="--no_aux_reward"; fi
ADV=advantage_inference/${TASK}_traj_tokens/${SUB}/${EXP}/adv_token_pred-gamma${GAMMA}-len${MAXLEN}-${EXP}.jsonl
