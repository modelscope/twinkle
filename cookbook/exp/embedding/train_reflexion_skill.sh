#!/bin/bash
# Online GRPO RFT for the reflexion skill generator (unified, self-contained, cached).
# GPUs: 8 — default high-memory layout uses rank 0 for actor training, rank 1 for
# the frozen ref model, ranks 2-3 for skill sampler (synced), and ranks 4-7 for
# base sampler (frozen). Override TRAIN_GPUS / REF_GPUS / SKILL_SAMPLER_GPUS /
# BASE_SAMPLER_GPUS for other layouts. Per chunk: base greedy
# solve -> rubric process-check (view A) ->
# skill-gen (thinking ON, N candidates) -> deterministic leak filter -> with-skill greedy
# pass -> group-relative advantage -> ONE on-policy GRPO step -> sync weights.
#
# Baseline rollouts + rubric diagnoses are disk-cached (output-dir/cache/*.jsonl), so a
# restart skips re-sampling them; skill-gen is on-policy and never cached. The next chunk's
# baseline is prefetched on a background thread (overlaps skill-gen; base sampler is frozen).
#
# The view-A rubric process-check uses the backup teacher API (set LLM_BACKUP_*). Without
# it the run still works: view A degrades to query-only and the leak filter stays
# deterministic (no teacher needed).

set -euo pipefail

export GEN_MODEL_ID=${GEN_MODEL_ID:-Qwen/Qwen3-4B}
export GEN_GPU_MEM=${GEN_GPU_MEM:-0.8}
# Datasets are pulled from ModelScope via twinkle.Dataset (ms://AI-MO/aops or
# ms://modelscope/competition_math); override AOPS_DATASET_ID / MATH_DATASET_ID to change.
# Teacher API for the view-A rubric process-check (optional; leak filter is deterministic).
export LLM_BACKUP_API_KEY=${LLM_BACKUP_API_KEY:-}
export LLM_BACKUP_BASE_URL=${LLM_BACKUP_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
export LLM_BACKUP_MODEL=${LLM_BACKUP_MODEL:-qwen3.7-max}

python cookbook/exp/embedding/train_reflexion_skill.py \
  --dataset aops \
  --n 5000 \
  --numeric-only \
  --chunk-size 32 \
  --n-skills 16 \
  --view-b-frac 0.5 \
  --xproblem-rubric \
  --skill-retries 2 \
  --balance \
  --balance-success-frac 0.3 \
  --balance-loop-frac 0.5 \
  --balance-max-draws-mult 8 \
  --max-tokens 8192 \
  --skill-max-tokens 4096 \
  --max-model-len 16384 \
  --eval-size 128 \
  --eval-every 5 \
  --sft-batch-size 8 \
  --ppo-mini-batch-size 0 \
  --grpo-epsilon 0.2 \
  --kl-beta 0.001 \
  --format-in-reward \
  --lr 1e-6 \
  --max-train-rounds 1500 \
  --save-rounds 200 \
  --trend-every 10 \
  --prefetch-baseline \
  --output-dir ./output/reflexion_skill \
  --swanlab-project twinkle \
  --swanlab-exp reflexion_skill_rft
