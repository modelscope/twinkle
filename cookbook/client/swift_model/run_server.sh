#!/usr/bin/env bash
set -euo pipefail

# Start a local Ray cluster (if needed) and launch the gemma-4-12B-it server.
# Override the GPU count with RAY_NUM_GPUS; the model app shards one 12B
# checkpoint with FSDP across 4 GPUs by default (see server_config.yaml).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

ray_port="${RAY_PORT:-6379}"
ray_address="${RAY_ADDRESS:-127.0.0.1:${ray_port}}"

if ! command -v ray >/dev/null 2>&1; then
  echo "ray is not installed; install the server dependencies first" >&2
  exit 1
fi
if ! command -v twinkle-server >/dev/null 2>&1; then
  echo "twinkle-server is not installed; run: pip install -e '.[client,server]'" >&2
  exit 1
fi
# The model app imports swift.dev.model.loader at actor startup, so ms-swift must
# be importable in this env.
if ! python -c 'import swift.dev.model.loader' >/dev/null 2>&1; then
  echo "ms-swift is not importable; from the ms-swift root run: pip install -e ." >&2
  exit 1
fi

if ! ray status --address="${ray_address}" >/dev/null 2>&1; then
  ray start \
    --head \
    --port="${ray_port}" \
    --num-gpus="${RAY_NUM_GPUS:-4}" \
    --include-dashboard=false \
    --disable-usage-stats
fi

config=cookbook/client/swift_model/server_config.yaml
twinkle-server check-config -c "${config}"
exec twinkle-server launch -c "${config}"
