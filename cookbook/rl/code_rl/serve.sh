#!/bin/sh
set -eu

# Start the OpenEnv code environment server for cookbook/rl/code_rl.
#
# This is the REMOTE side: run it on any machine reachable from the training
# driver — no Docker, no KVM, no GPU. The training script only needs the
# resulting URL.
#
# One-time setup:
#   pip install openenv
#   pip install -e /path/to/OpenEnv/envs/coding_env
#
# Usage:
#   sh serve.sh                      # 4 workers x 64 sessions = 256 sessions
#   WORKERS=8 PORT=9000 sh serve.sh
#   HOST=127.0.0.1 sh serve.sh       # same-machine training: no network exposure
#
# SECURITY: OpenEnv has no authentication. Anyone who can reach this port can
# execute code in the sandbox and consume your capacity. Either bind to
# 127.0.0.1 (when training runs on this same host), or restrict the port to the
# training host's IP with a security group / firewall rule. Never leave the
# default 0.0.0.0 reachable from the public internet.
#
# Then point the training script at it:
#   OPENENV_BASE_URL=http://<this-host>:8000 sh run_openenv.sh

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Uvicorn worker processes. Each worker holds its own app instance, so total
# capacity is WORKERS x MAX_CONCURRENT_ENVS. Size it above the number of
# concurrent trajectories (BATCH_SIZE x NUM_GENERATIONS), otherwise the extra
# WebSocket connections are rejected at capacity.
WORKERS="${WORKERS:-4}"
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-64}"

cd "$(dirname "$0")"

echo "Serving twinkle_code_env on ${HOST}:${PORT}"
echo "  workers=${WORKERS}, max_concurrent_envs=${MAX_CONCURRENT_ENVS} per worker"
echo "  capacity=$((WORKERS * MAX_CONCURRENT_ENVS)) concurrent sessions"
if [ "$HOST" = "0.0.0.0" ]; then
    echo "  WARNING: bound to 0.0.0.0 with no authentication — restrict this port"
    echo "           to the training host, or use HOST=127.0.0.1 if training is local."
fi

exec uvicorn server_app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

# Docker alternative (build coding_env's image, then override the app module):
#   docker build -t coding-env:latest -f /path/to/OpenEnv/envs/coding_env/server/Dockerfile /path/to/OpenEnv
# Note the image serves upstream coding_env.server.app, which is capped at one
# session; mount this file and point uvicorn at it to keep the concurrency fix.
