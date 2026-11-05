#!/usr/bin/env bash
# .139 policy 评估：单卡（卡3），200 题，greedy，iterative1024 分块 rollout
# policy 用 LoRA adapter（policy_chunked8192）加载到 base 模型上
# 启动前等待卡 3 空闲（防抢），支持 termination_mode 切换
set -euo pipefail

ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
MODEL=/data3/xhy/codes/models/Qwen3-8B
RUN=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
ADAPTER=$RUN/policy_chunked8192
INPUT=$ROOT/data/qa/olympiadbench_eval200_seed20261031.jsonl
OUT=$ROOT/data/qa/olympiad_eval200_seed20261031
TMP=/data3/xhy/tmp/qevolve_eval

# 评测协议：strict 与 base 同口径；若 finished 率过低可改 consensus
MODE=${1:-strict}
OFFSET=${OFFSET:-0}
LIMIT=${LIMIT:-200}
WALL_CLOCK=${WALL_CLOCK:-2400}
GPU=3
FREE_NEED=40000   # 卡3 需要 >40G 空闲（8B + 长上下文）

mkdir -p "$OUT" "$TMP"
export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$TMP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

# 前置断言
for f in "$INPUT" "$MODEL/config.json" "$ADAPTER/adapter_config.json" "$ROOT/eval_agent/qa_rollout_chunked_context.py"; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

# 等待卡 3 空闲（防抢）
echo "$(date '+%F %T') 等待卡${GPU}空闲（需 free>${FREE_NEED}MiB）..."
while :; do
  free=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "${free:-0}" -ge "$FREE_NEED" ]; then
    echo "$(date '+%F %T') 卡${GPU}空闲 ${free}MiB，开始 policy 评估"
    break
  fi
  sleep 60
done

# 清理旧结果
if [ "$OFFSET" = "0" ]; then
  rm -f "$OUT/policy.jsonl" "$OUT/policy.log" "$OUT/policy_status" "$OUT/summary_policy.json"
else
  echo "resume mode: offset=$OFFSET, keep existing policy.jsonl"
fi

# 单卡跑全量 200 题
echo "$(date '+%F %T') 启动 policy 评估（mode=$MODE, 200题, GPU$GPU）"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u "$ROOT/eval_agent/qa_rollout_chunked_context.py" \
  --input "$INPUT" --output "$OUT/policy.jsonl" \
  --model "$MODEL" --adapter "$ADAPTER" \
  --offset "$OFFSET" --limit "$LIMIT" --samples 1 \
  --max_new_tokens 8192 --step_max_new_tokens 1024 --max_steps 512 \
  --context_max_tokens 32768 --temperature 0 --termination_mode "$MODE" \
  --append --wall_clock_timeout "$WALL_CLOCK" \
  >"$OUT/policy.log" 2>&1
rc=$?
echo "$rc" > "$OUT/policy_status"
echo "$(date '+%F %T') policy 评估结束 rc=$rc，产出 $(wc -l < "$OUT/policy.jsonl" 2>/dev/null || echo 0) 条"

# 统计（与 base 同口径）
"$PY" - "$OUT" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
f = p / "policy.jsonl"
if not f.exists():
    print("NO_RESULT"); sys.exit(0)
rows = [json.loads(x) for x in f.open() if x.strip()]
n = len(rows)
if n == 0:
    print("EMPTY_RESULT"); sys.exit(0)
ok = sum(bool(r.get('meta', {}).get('success', False)) for r in rows)
fin = sum(bool(r.get('meta', {}).get('finished', False)) for r in rows)
steps = [len(r.get('conversations') or []) for r in rows]
rep = {'model': 'policy_chunked8192', 'n': n, 'correct': ok,
       'accuracy': round(ok / n, 4), 'finished': fin,
       'finished_rate': round(fin / n, 4),
       'turns_mean': round(sum(steps) / len(steps), 1)}
(p / 'summary_policy.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2))
print(json.dumps(rep, ensure_ascii=False, indent=2))
PYEOF

exit "$rc"
