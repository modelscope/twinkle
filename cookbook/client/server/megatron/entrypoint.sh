#!/bin/bash

# Container entrypoint for Twinkle Megatron service.
# This process only supervises run.sh itself. Service health and Ray/Twinkle
# restarts are handled inside run.sh.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
RESTART_BACKOFF_SECONDS=10

CHILD_PID=""

stop_child_and_exit() {
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 143
}

trap stop_child_and_exit TERM INT

case "${1:-}" in
    --help|-h|--restart)
        exec bash "$RUN_SCRIPT" "$@"
        ;;
esac

while true; do
    bash "$RUN_SCRIPT" "$@" &
    CHILD_PID=$!

    wait "$CHILD_PID"
    EXIT_CODE=$?
    CHILD_PID=""

    echo "[twinkle-entrypoint] run.sh exited with code $EXIT_CODE; restarting in ${RESTART_BACKOFF_SECONDS}s"
    sleep "$RESTART_BACKOFF_SECONDS" &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
done
