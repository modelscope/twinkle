#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

: "${MODEL_ID:?Set MODEL_ID to a Hugging Face-compatible model directory or model ID}"
: "${TENANT_A_DATASET_ID:?Set TENANT_A_DATASET_ID to a GSM8K dataset path or ID}"
: "${TENANT_B_DATASET_ID:?Set TENANT_B_DATASET_ID to a GSM8K dataset path or ID}"

ray_port="${RAY_PORT:-6379}"
ray_address="${RAY_ADDRESS:-127.0.0.1:${ray_port}}"
ray_num_gpus="${RAY_NUM_GPUS:-3}"

if ! command -v ray >/dev/null 2>&1; then
  echo "ray is not installed; install the async-RL dependencies first" >&2
  exit 1
fi

if ! ray status --address="${ray_address}" >/dev/null 2>&1; then
  ray start \
    --head \
    --port="${ray_port}" \
    --num-gpus="${ray_num_gpus}" \
    --include-dashboard=false \
    --disable-usage-stats
fi

exec python cookbook/rl/async_rl/async_multi_lora_grpo.py \
  --config cookbook/rl/async_rl/async_multi_lora_grpo.yaml
