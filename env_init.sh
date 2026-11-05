#!/bin/bash
# Environment bootstrap for Snellius (SURF) + SLURM, sourced by the run scripts
# from the repo root. Override any path below via environment variables.

source ~/.bashrc

# ---- GPU / build toolchain (Snellius modules) ----
if command -v module >/dev/null 2>&1; then
  module purge
  module load 2024
  module load CUDA/12.6.0
  module load GLib/2.80.4-GCCcore-13.3.0
  module load GCCcore/13.3.0
  module load Java/17
fi

# ---- Python env (single conda env with sglang + pyserini + vllm + trl) ----
conda activate qevolve
export PYTHON="${PYTHON:-${HOME}/anaconda3/envs/qevolve/bin/python}"

# ---- torchrun master port (avoid EADDRINUSE across concurrent Slurm jobs) ----
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 55536))}"
else
  export MASTER_PORT="${MASTER_PORT:-$((29500 + ($$ % 1000)))}"
fi

# ---- Storage roots ----
# PRJS       : shared project storage (base models, checkpoints, env data)
# SCRATCH_ROOT: run outputs (QEvolve scripts write ${SCRATCH_ROOT}/QLASS-log, /QLASS-Adv)
# SCRATCH    : legacy alias used by some legacy scripts
export PRJS="${PRJS:-/projects/prjs0880/QLASS}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/yzhang2}"
export SCRATCH="${SCRATCH:-${SCRATCH_ROOT}/QLASS-log}"

# ---- Weights & Biases ----
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/QLASS-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-QLASS}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ---- Hugging Face cache ----
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/.cache/huggingface}"

# ---- Environment data (webshop / alfworld) ----
export ALFWORLD_DATA="${ALFWORLD_DATA:-/projects/prjs0880/QLASS/env_data/alfworld}"
export WEBSHOP_DATA="${WEBSHOP_DATA:-/projects/prjs0880/QLASS/env_data/webshop}"
