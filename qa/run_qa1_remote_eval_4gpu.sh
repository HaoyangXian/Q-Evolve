#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xhy/codes/work/paper_repos/Q-Evolve
PYTHON=/data/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data/xhy/codes/models/Qwen3-4B
INPUT=${ROOT}/data/qa/gsm8k_eval150.jsonl
ADAPTER=${ROOT}/runs/qa1_fixed_policy_4096
OUT=${ROOT}/runs/qa1_fixed_eval
mkdir -p "${OUT}"
export PYTHONPATH=${ROOT}:${ROOT}/QEvolve
export TOKENIZERS_PARALLELISM=false

run_shard() {
  local gpu="$1" offset="$2" limit="$3" tag="$4" adapter="$5"
  local extra=()
  if [ "${adapter}" = "1" ]; then extra+=(--adapter "${ADAPTER}"); fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/eval_agent/qa_rollout.py" \
    --input "${INPUT}" --output "${OUT}/${tag}.jsonl" \
    --offset "${offset}" --limit "${limit}" --samples 1 \
    --model "${MODEL}" --max_new_tokens 512 --temperature 0 "${extra[@]}" \
    > "${OUT}/${tag}.log" 2>&1
}

run_shard 0 0   38 base0 0 & p1=$!
run_shard 1 38  38 base1 0 & p2=$!
run_shard 2 76  38 base2 0 & p3=$!
run_shard 3 114 36 base3 0 & p4=$!
run_shard 4 0   38 policy0 1 & p5=$!
run_shard 5 38  38 policy1 1 & p6=$!
run_shard 6 76  38 policy2 1 & p7=$!
run_shard 7 114 36 policy3 1 & p8=$!
wait "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7" "$p8"

cat "${OUT}/base0.jsonl" "${OUT}/base1.jsonl" "${OUT}/base2.jsonl" "${OUT}/base3.jsonl" > "${OUT}/gsm8k_eval150_base.jsonl"
cat "${OUT}/policy0.jsonl" "${OUT}/policy1.jsonl" "${OUT}/policy2.jsonl" "${OUT}/policy3.jsonl" > "${OUT}/gsm8k_eval150_policy.jsonl"

"${PYTHON}" - <<'PY'
import json, os
from eval_agent.qa_verifier import answers_equal, has_final_answer
root = "/data/xhy/codes/work/paper_repos/Q-Evolve/runs/qa1_fixed_eval"
for tag in ("base", "policy"):
    path = os.path.join(root, f"gsm8k_eval150_{tag}.jsonl")
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    ok = 0
    finished = 0
    for row in rows:
        raw = row.get("conversations", [])[-1].get("value", "") if row.get("conversations") else ""
        finished += int(has_final_answer(raw))
        ok += int(has_final_answer(raw) and answers_equal(row.get("pred_answer", ""), row.get("gold_answer", "")))
    print(json.dumps({"model": tag, "n": len(rows), "finished": finished, "correct": ok, "exact_match": ok / len(rows) if rows else 0.0}))
PY
