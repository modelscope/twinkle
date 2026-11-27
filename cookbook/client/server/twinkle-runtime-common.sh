#!/usr/bin/env bash

# The platform restarts the workload when its foreground command exits.
pids=()
services=()
work_dir=

cleanup() {
    trap '' INT TERM
    local index pid
    for ((index=${#pids[@]} - 1; index >= 0; index--)); do
        pid=${pids[index]}
        if jobs -p | grep -Fxq "$pid"; then
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$work_dir" ]]; then
        rm -r -- "$work_dir"
    fi
}

trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

load_config() {
    set -a
    # The deployment operator owns this shell configuration file.
    source "$1"
    set +a
}

start_service() {
    "$@" <&0 &
    pids+=("$!")
    services+=("${1##*/}")
}

wait_for_services() {
    local result=0 finished='' index name=unknown pid
    for pid in "${pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            finished=$pid
            wait "$pid" || result=$?
            break
        fi
    done
    if [[ -z "$finished" ]]; then
        wait -n -p finished "${pids[@]}" || result=$?
    fi
    for index in "${!pids[@]}"; do
        if [[ "${pids[index]}" == "${finished:-}" ]]; then
            name=${services[index]}
            break
        fi
    done
    printf 'Service %s exited (status %s).\n' "$name" "$result" >&2
    return "$((result == 0 ? 1 : result))"
}
