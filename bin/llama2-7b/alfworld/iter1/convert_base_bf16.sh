#!/bin/bash
# One-time: re-save the fp32 SFT base as bf16 (26GB -> 13GB) so every run loads it
# ~2x faster. Usable by critic (AutoModel reads the backbone) and policy
# (AutoModelForCausalLM). CPU only, no GPU.
source env_init.sh

SRC=${SRC:-eval_agent/qlass-Llama-2-7b-chat-hf-alfworld-sft}
DST=${DST:-$HOME/models/qlass-Llama-2-7b-chat-hf-alfworld-sft-bf16}

$PYTHON - <<PY
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
src = "${SRC}"; dst = "${DST}"
print(f"[convert] loading {src} as bf16 (low_cpu_mem_usage)...", flush=True)
m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print(f"[convert] saving bf16 to {dst} ...", flush=True)
os.makedirs(dst, exist_ok=True)
m.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained(src).save_pretrained(dst)
print("[convert] DONE", flush=True)
PY
echo "CONVERT_EXIT=$?"
