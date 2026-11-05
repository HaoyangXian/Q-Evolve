#!/usr/bin/env bash
# sft3 评测 v2：4 卡（4,5,6,7），vLLM continuous batching + prefix cache 复用
# 每卡一个 vLLM 离线引擎，维护 --concurrency 道活跃题；200 题切 4 片（各 50）
# greedy (temp 0)，逐 step 生成，终止标记/上限与旧版完全一致
set -euo pipefail

ROOT=/data3/xhy/codes/work/paper_repos/Q-Evolve
PY=/data3/xhy/miniconda3/envs/pettingllms/bin/python
SFT_MODEL=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small/sft_success
INPUT=$ROOT/data/qa/olympiadbench_eval200_seed20261031.jsonl
OUT=$ROOT/data/qa/olympiad_eval200_seed20261031
RUNLOG=$ROOT/runs/olympiad_qwen3_8b_iterative1024_small
RES=$ROOT/bin/qa/gpu_reserve.py
TMP=/data3/xhy/tmp/qevolve_eval
mkdir -p "$OUT" "$TMP" "$RUNLOG"

export PYTHONPATH="$ROOT:$ROOT/QEvolve"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$TMP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=(4 5 6 7)
OFFS=(0 50 100 150)
LIMS=(50 50 50 50)
N=${#GPUS[@]}
PREFIX=sft3
CONCURRENCY=6          # 每卡同时维护的活跃题数
GPU_MEM_UTIL=0.88      # 显存预算，防爆显存

start_reserve() {
  local gpu="$1"
  nohup "$PY" "$RES" --gpus "$gpu" --fraction 0.9 >"/tmp/qevolve_${gpu}_reserve.log" 2>&1 </dev/null &
  echo $!
}
stop_reserve() {
  local p="$1"
  kill -TERM "$p" 2>/dev/null || true
  sleep 1
  kill -KILL "$p" 2>/dev/null || true
}

# ---------- 前置断言 ----------
for f in "$INPUT" "$SFT_MODEL/config.json" "$ROOT/eval_agent/qa_rollout_vllm_batched.py"; do
  [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done
for i in $(seq 0 $((N-1))); do
  free=$(nvidia-smi --id="${GPUS[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [ "${free:-0}" -ge 30000 ] || { echo "GPU${GPUS[$i]} only ${free}MiB free (<30000)" >&2; exit 3; }
done
echo "$(date '+%F %T') preflight OK: 4 GPUs free, input=$(wc -l < "$INPUT") lines"

# ---------- 清旧分片 ----------
rm -f "$OUT/${PREFIX}_part"*.jsonl "$OUT/${PREFIX}_part"*.log "$OUT/${PREFIX}_status"* "$OUT/sft.jsonl" "$OUT/SFT3_EVAL_DONE"

# ---------- 启动 4 个分片（错开 30s，避免同时加载 31G 权重抢 IO）----------
declare -a pids=() flags=() reserves=()
for i in $(seq 0 $((N-1))); do
  (
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -u "$ROOT/eval_agent/qa_rollout_vllm_batched.py" \
      --input "$INPUT" --output "$OUT/${PREFIX}_part${i}.jsonl" --model "$SFT_MODEL" \
      --offset "${OFFS[$i]}" --limit "${LIMS[$i]}" --samples 1 \
      --step_max_new_tokens 1024 --max_steps 512 \
      --context_max_tokens 32768 --temperature 0 --termination_mode strict \
      --concurrency "$CONCURRENCY" --gpu_memory_utilization "$GPU_MEM_UTIL" \
      >"$OUT/${PREFIX}_part${i}.log" 2>&1
    echo $? > "$OUT/${PREFIX}_status${i}"
  ) &
  pids+=("$!")
  flags+=(0)
  echo "$(date '+%T') launched part$i GPU${GPUS[$i]} offset=${OFFS[$i]} limit=${LIMS[$i]} concurrency=${CONCURRENCY} pid=$!"
  [ "$i" -lt $((N-1)) ] && sleep 30
done

# ---------- 等待 ----------
rc_total=0
while :; do
  all=1
  for i in $(seq 0 $((N-1))); do
    [ "${flags[$i]}" -eq 1 ] && continue
    all=0
    if [ -f "$OUT/${PREFIX}_status${i}" ]; then
      rc=$(cat "$OUT/${PREFIX}_status${i}" 2>/dev/null || echo 99)
      got=$(wc -l < "$OUT/${PREFIX}_part${i}.jsonl" 2>/dev/null || echo 0)
      if [ "$rc" -eq 0 ] && [ "$got" -ge "${LIMS[$i]}" ]; then
        flags[$i]=1
        rp=$(start_reserve "${GPUS[$i]}")
        reserves+=("$rp")
        echo "$(date '+%T') part$i DONE ($got/${LIMS[$i]}) -> GPU${GPUS[$i]} reserved pid=$rp"
      elif [ "$rc" -ne 0 ]; then
        echo "$(date '+%T') part$i FAILED rc=$rc (got $got/${LIMS[$i]}), see ${PREFIX}_part${i}.log" >&2
        flags[$i]=1
        rc_total=$rc
      fi
    fi
  done
  [ "$all" -eq 1 ] && break
  sleep 10
done

# ---------- 合并 ----------
for i in $(seq 0 $((N-1))); do
  [ -s "$OUT/${PREFIX}_part${i}.jsonl" ] && cat "$OUT/${PREFIX}_part${i}.jsonl"
done > "$OUT/sft.jsonl"
echo "$(date '+%T') merged sft.jsonl = $(wc -l < "$OUT/sft.jsonl") lines"

# ---------- 统计 ----------
"$PY" - "$OUT" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
def stat(name):
    f = p / f"{name}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(x) for x in f.open() if x.strip()]
    n = len(rows)
    if n == 0:
        return None
    ok = sum(bool(r.get('meta', {}).get('success', False)) for r in rows)
    fin = sum(bool(r.get('meta', {}).get('finished', False)) for r in rows)
    steps = [len(r.get('conversations') or []) for r in rows]
    return {'n': n, 'correct': ok, 'accuracy': round(ok / n, 4),
            'finished': fin, 'finished_rate': round(fin / n, 4),
            'turns_mean': round(sum(steps) / len(steps), 1)}
rep = {'base': stat('base'), 'sft': stat('sft')}
if rep['base'] and rep['sft']:
    rep['sft_minus_base_pt'] = round((rep['sft']['accuracy'] - rep['base']['accuracy']) * 100, 2)
    rep['sft_minus_base_finished_pt'] = round((rep['sft']['finished_rate'] - rep['base']['finished_rate']) * 100, 2)
(p / 'summary_sft3_4gpu.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2))
print(json.dumps(rep, ensure_ascii=False, indent=2))
PYEOF

# ---------- 释放 reserve ----------
for p in "${reserves[@]}"; do stop_reserve "$p"; done
echo "$(date '+%F %T') sft3 eval finished rc=$rc_total"

if [ "$rc_total" -eq 0 ]; then touch "$OUT/SFT3_EVAL_DONE"; fi
exit "$rc_total"
