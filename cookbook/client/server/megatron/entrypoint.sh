#!/bin/bash

# Container entrypoint for Twinkle Megatron service.
# This process supervises run.sh and owns external health checks.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
TWINKLE_WORK_DIR="${TWINKLE_WORK_DIR:-/dashscope/caches/application/twinkle}"
TEMP_DIR="${TWINKLE_TEMP_DIR:-/dashscope/caches/application/ray_logs}"
LOG_FILE="$TWINKLE_WORK_DIR/run.log"
TWINKLE_HEALTH_URL="${TWINKLE_HEALTH_URL:-http://127.0.0.1:9000/api/v1/healthz}"
TWINKLE_DEEP_HEALTH_URL="${TWINKLE_DEEP_HEALTH_URL:-http://127.0.0.1:9000/api/v1/twinkle/healthz/deep}"
TWINKLE_WATCHDOG_INTERVAL_SECONDS="${TWINKLE_WATCHDOG_INTERVAL_SECONDS:-10}"
TWINKLE_WATCHDOG_FAILURE_THRESHOLD="${TWINKLE_WATCHDOG_FAILURE_THRESHOLD:-3}"
TWINKLE_RAY_GRACE_SECONDS="${TWINKLE_RAY_GRACE_SECONDS:-30}"
TWINKLE_HEALTH_GRACE_SECONDS="${TWINKLE_HEALTH_GRACE_SECONDS:-${TWINKLE_WATCHDOG_STARTUP_GRACE_SECONDS:-300}}"
TWINKLE_DEEP_HEALTH_GRACE_SECONDS="${TWINKLE_DEEP_HEALTH_GRACE_SECONDS:-${TWINKLE_HEALTH_GRACE_SECONDS:-300}}"
RESTART_BACKOFF_SECONDS="${TWINKLE_ENTRYPOINT_RESTART_BACKOFF_SECONDS:-10}"

CHILD_PID=""

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

require_non_negative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        print_error "$name 必须是非负整数，当前值: $value"
        exit 1
    fi
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        print_error "$name 必须是正整数，当前值: $value"
        exit 1
    fi
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" &> /dev/null; then
        print_error "缺少必需命令: $command_name"
        exit 1
    fi
}

validate_entrypoint_config() {
    require_positive_int "TWINKLE_WATCHDOG_INTERVAL_SECONDS" "$TWINKLE_WATCHDOG_INTERVAL_SECONDS"
    require_positive_int "TWINKLE_WATCHDOG_FAILURE_THRESHOLD" "$TWINKLE_WATCHDOG_FAILURE_THRESHOLD"
    require_non_negative_int "TWINKLE_RAY_GRACE_SECONDS" "$TWINKLE_RAY_GRACE_SECONDS"
    require_non_negative_int "TWINKLE_HEALTH_GRACE_SECONDS" "$TWINKLE_HEALTH_GRACE_SECONDS"
    require_non_negative_int "TWINKLE_DEEP_HEALTH_GRACE_SECONDS" "$TWINKLE_DEEP_HEALTH_GRACE_SECONDS"
    require_non_negative_int "TWINKLE_ENTRYPOINT_RESTART_BACKOFF_SECONDS" "$RESTART_BACKOFF_SECONDS"

    require_command timeout
    require_command ray
    require_command tail

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null \
        && ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "缺少 HTTP health 检查命令: curl, wget, python3 或 python"
        exit 1
    fi
}

check_http_health() {
    local url="${1:-$TWINKLE_HEALTH_URL}"
    if command -v curl &> /dev/null; then
        curl -fsS --max-time 10 "$url" >/dev/null
        return
    fi

    if command -v wget &> /dev/null; then
        wget -q -O /dev/null --timeout=10 "$url"
        return
    fi

    local python_bin="python3"
    if ! command -v "$python_bin" &> /dev/null; then
        python_bin="python"
    fi
    "$python_bin" - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
}

print_watchdog_diagnostics() {
    print_warning "EntryPoint watchdog 诊断信息："
    echo "  - health url: $TWINKLE_HEALTH_URL"
    echo "  - deep health url: $TWINKLE_DEEP_HEALTH_URL"
    echo "  - run.sh pid: ${CHILD_PID:-unset}"
    echo "  - Ray logs: $TEMP_DIR/session_latest/logs"

    print_warning "ray status 输出："
    ray status 2>&1 || true

    if [ -f "$LOG_FILE" ]; then
        print_warning "最近 100 行 Twinkle Server 日志："
        tail -n 100 "$LOG_FILE" || true
    fi
}

stop_child() {
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    CHILD_PID=""
}

stop_child_and_exit() {
    stop_child
    exit 143
}

trap stop_child_and_exit TERM INT

case "${1:-}" in
    --help|-h|--restart)
        exec bash "$RUN_SCRIPT" "$@"
        ;;
esac

validate_entrypoint_config

while true; do
    TWINKLE_RUN_EXISTING_ACTION="${TWINKLE_RUN_EXISTING_ACTION:-restart}" bash "$RUN_SCRIPT" "$@" &
    CHILD_PID=$!

    WATCHDOG_FAILURES=0
    WATCHDOG_STARTED_AT=$SECONDS
    EXIT_CODE=0

    while true; do
        if ! kill -0 "$CHILD_PID" 2>/dev/null; then
            wait "$CHILD_PID"
            EXIT_CODE=$?
            CHILD_PID=""
            break
        fi

        WATCHDOG_FAILURE_REASON=""
        WATCHDOG_GRACE_SECONDS=0
        if ! timeout 10 ray status >/dev/null 2>&1; then
            WATCHDOG_FAILURE_REASON="ray status failed"
            WATCHDOG_GRACE_SECONDS="$TWINKLE_RAY_GRACE_SECONDS"
        elif ! check_http_health; then
            WATCHDOG_FAILURE_REASON="http health check failed: $TWINKLE_HEALTH_URL"
            WATCHDOG_GRACE_SECONDS="$TWINKLE_HEALTH_GRACE_SECONDS"
        elif ! check_http_health "$TWINKLE_DEEP_HEALTH_URL"; then
            WATCHDOG_FAILURE_REASON="deep health check failed (model actors may be dead): $TWINKLE_DEEP_HEALTH_URL"
            WATCHDOG_GRACE_SECONDS="$TWINKLE_DEEP_HEALTH_GRACE_SECONDS"
        fi

        if [ -z "$WATCHDOG_FAILURE_REASON" ]; then
            WATCHDOG_FAILURES=0
        else
            WATCHDOG_ELAPSED=$(( SECONDS - WATCHDOG_STARTED_AT ))
            if [ "$WATCHDOG_ELAPSED" -lt "$WATCHDOG_GRACE_SECONDS" ]; then
                print_warning "EntryPoint watchdog 启动宽限期内检查失败 (${WATCHDOG_ELAPSED}s/${WATCHDOG_GRACE_SECONDS}s): $WATCHDOG_FAILURE_REASON"
            else
                WATCHDOG_FAILURES=$((WATCHDOG_FAILURES + 1))
                print_warning "EntryPoint watchdog 检查失败 ($WATCHDOG_FAILURES/$TWINKLE_WATCHDOG_FAILURE_THRESHOLD): $WATCHDOG_FAILURE_REASON"
            fi
        fi

        if [ "$WATCHDOG_FAILURES" -ge "$TWINKLE_WATCHDOG_FAILURE_THRESHOLD" ]; then
            print_error "EntryPoint watchdog 连续失败达到阈值，准备重启 run.sh"
            print_watchdog_diagnostics
            stop_child
            EXIT_CODE=1
            break
        fi

        sleep "$TWINKLE_WATCHDOG_INTERVAL_SECONDS"
    done

    echo "[twinkle-entrypoint] run.sh exited with code $EXIT_CODE; restarting in ${RESTART_BACKOFF_SECONDS}s"
    sleep "$RESTART_BACKOFF_SECONDS" &
    CHILD_PID=$!
    wait "$CHILD_PID" 2>/dev/null || true
    CHILD_PID=""
done
