#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

: "${TWINKLE_LOCAL_MODEL_PATH:?Set TWINKLE_LOCAL_MODEL_PATH to the local Qwen3.5-4B directory}"

ray_port="${RAY_PORT:-6379}"
ray_address="${RAY_ADDRESS:-127.0.0.1:${ray_port}}"

if ! command -v ray >/dev/null 2>&1; then
  echo "ray is not installed; install the async-RL dependencies first" >&2
  exit 1
fi
if ! command -v twinkle-server >/dev/null 2>&1; then
  echo "twinkle-server is not installed; run: pip install -e '.[async-rl,client]'" >&2
  exit 1
fi

if ! ray status --address="${ray_address}" >/dev/null 2>&1; then
  ray start \
    --head \
    --port="${ray_port}" \
    --num-gpus="${RAY_NUM_GPUS:-2}" \
    --include-dashboard=false \
    --disable-usage-stats
fi

config=cookbook/client/async_rl/server_config.yaml
twinkle-server check-config -c "${config}"
exec twinkle-server launch -c "${config}"
