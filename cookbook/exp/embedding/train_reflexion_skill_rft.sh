#!/bin/bash
# RFT cold-start for the reflexion skill generator (see reflexion.md §6).
# GPUs: 8 — ranks 0-3 train (skill model, FSDP2), 4-5 skill sampler, 6-7 base sampler.
# Leak filtering uses the backup teacher API (no local judge): set LLM_BACKUP_*.

set -euo pipefail

export GEN_MODEL_ID=${GEN_MODEL_ID:-Qwen/Qwen3.5-4B}
export MATH_DATA_DIR=${MATH_DATA_DIR:-./output/math_data/MATH}
export LLM_BACKUP_API_KEY=${LLM_BACKUP_API_KEY:?set LLM_BACKUP_API_KEY for the leak judge}
export LLM_BACKUP_BASE_URL=${LLM_BACKUP_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export LLM_BACKUP_MODEL=${LLM_BACKUP_MODEL:-qwen3.7-max}

python cookbook/exp/embedding/train_reflexion_skill_rft.py \
  --n 2000 \
  --chunk-size 16 \
  --n-skills 8 \
  --pass-k 8 \
  --hard-baseline-max 0.25 \
  --train-every 64 \
  --train-batch 64 \
  --sft-batch-size 8 \
  --lr 1e-5 \
  --max-train-rounds 200 \
  --output-dir ./output/reflexion_skill_rft
