#!/bin/bash
# RFT cold-start for the reflexion skill generator (see reflexion.md §6).
# GPUs: 8 — ranks 0-3 train (skill model, FSDP2), 4-5 skill sampler, 6-7 base sampler.
# Leak filtering uses the backup teacher API (no local judge): set LLM_BACKUP_*.

set -euo pipefail

export GEN_MODEL_ID=${GEN_MODEL_ID:-Qwen/Qwen3-4B}
# Local MATH copy (modelscope download cache). Override MATH_DATA_DIR if the
# cache hash dir changes or the data lives elsewhere.
export MATH_DATA_DIR=${MATH_DATA_DIR:-/mnt/workspace/.cache/modelscope/hub/datasets/downloads/extracted/0744cd2d347a7e8f85f7087d950b2ed38b626a5c808c5399e2d8a0923d42d013/MATH}
export LLM_BACKUP_API_KEY=${LLM_BACKUP_API_KEY:?set LLM_BACKUP_API_KEY for the leak judge}
export LLM_BACKUP_BASE_URL=${LLM_BACKUP_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export LLM_BACKUP_MODEL=${LLM_BACKUP_MODEL:-qwen3.7-max}

python cookbook/exp/embedding/train_reflexion_skill_rft.py \
  --n 5000 \
  --chunk-size 16 \
  --n-skills 8 \
  --view-b-frac 0.5 \
  --skill-retries 2 \
  --max-tokens 25000 \
  --max-model-len 30000 \
  --sft-batch-size 8 \
  --grpo-epsilon 0.2 \
  --lr 6e-6 \
  --max-train-rounds 1500 \
  --save-rounds 25 \
  --trend-every 10 \
  --output-dir ./output/reflexion_skill_rft
