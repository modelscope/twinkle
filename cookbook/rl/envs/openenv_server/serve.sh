#!/bin/sh
set -eu
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
