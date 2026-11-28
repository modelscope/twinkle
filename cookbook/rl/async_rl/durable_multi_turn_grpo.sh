#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-ms://Qwen/Qwen3.5-4B}"
MODEL_GPUS="${MODEL_GPUS:-4}"
SAMPLER_GPUS="${SAMPLER_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-100}"

python cookbook/rl/async_rl/durable_multi_turn_grpo.py \
  --model-id "${MODEL_ID}" \
  --model-gpus "${MODEL_GPUS}" \
  --sampler-gpus "${SAMPLER_GPUS}" \
  --max-steps "${MAX_STEPS}" \
  "$@"
