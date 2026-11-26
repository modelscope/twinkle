#!/usr/bin/env bash
set -euo pipefail

export FLA_TILELANG="${FLA_TILELANG:-0}"

start_worker() {
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address (host:port)}"
    exec ray start --address="$RAY_ADDRESS" --block "$@"
}

start_head() {
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address (host:port)}"
    : "${TWINKLE_SERVER_CONFIG:?Set TWINKLE_SERVER_CONFIG to the server YAML path}"
    python -m twinkle.server check-config --config "$TWINKLE_SERVER_CONFIG"

    source "$(dirname -- "${BASH_SOURCE[0]}")/twinkle-runtime-common.sh"
    local finished readiness_pid

    start_service ray start --head --block "$@"
    start_service timeout "${TWINKLE_CLUSTER_START_TIMEOUT:-600}" python -u - <<'PYTHON'
import os
import time

import ray

expected_nodes = int(os.environ.get("TWINKLE_EXPECTED_RAY_NODES", "3"))
gpus_per_node = int(os.environ.get("TWINKLE_EXPECTED_GPUS_PER_NODE", "8"))
print(f"Waiting for {expected_nodes} Ray nodes with {gpus_per_node} GPUs each.")
address = os.environ["RAY_ADDRESS"]
while True:
    try:
        # GCS can accept connections before the local Ray node registers.
        ray.init(address=address, log_to_driver=False)
        break
    except ConnectionError:
        time.sleep(2)
try:
    while True:
        counts = [node["Resources"].get("GPU", 0) for node in ray.nodes() if node["Alive"]]
        if counts == [gpus_per_node] * expected_nodes:
            print(f"Ray ready: {expected_nodes} nodes, {sum(counts):g} GPUs.")
            break
        time.sleep(2)
finally:
    ray.shutdown()
PYTHON
    readiness_pid=$!
    if ! wait -n -p finished "${pids[@]}" || [[ "${finished:-}" != "$readiness_pid" ]]; then
        printf 'Ray cluster did not reach its expected capacity before startup.\n' >&2
        exit 1
    fi
    unset 'pids[1]' 'services[1]'

    start_service python -m twinkle.server launch --config "$TWINKLE_SERVER_CONFIG"
    wait_for_services
}

role=${1:-}
if [ "$#" -gt 0 ]; then
    shift
fi

case "$role" in
    h20-a|h20-b|h20-c)
        : "${NODE_IP:?Set NODE_IP to the private IPv4 address of this node}"
        : "${TWINKLE_DEFAULT_SAVE_DIR:?Set the shared checkpoint directory}"
        if [ ! -d "$TWINKLE_DEFAULT_SAVE_DIR" ] || [ ! -w "$TWINKLE_DEFAULT_SAVE_DIR" ]; then
            printf 'Checkpoint directory must exist and be writable: %s\n' "$TWINKLE_DEFAULT_SAVE_DIR" >&2
            exit 1
        fi
        node=${role#h20-}
        options=(--node-ip-address="$NODE_IP" "--labels=twinkle-node=$node"
                 --disable-usage-stats)
        if [ "$node" = a ]; then
            export RAY_ADDRESS="$NODE_IP:6379"
            export TWINKLE_EXPECTED_RAY_NODES=3 TWINKLE_EXPECTED_GPUS_PER_NODE=8
            export TWINKLE_SERVER_CONFIG="${TWINKLE_SERVER_CONFIG:-/twinkle/cookbook/client/server/megatron/server_config_h20_3x8.yaml}"
            start_head --port=6379 --dashboard-host=127.0.0.1 "${options[@]}" "$@"
        else
            start_worker "${options[@]}" "$@"
        fi
        ;;
    head)
        start_head "$@"
        ;;
    ray-head)
        exec ray start --head --block "$@"
        ;;
    ray-worker)
        start_worker "$@"
        ;;
    server)
        : "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray head address (host:port)}"
        exec python -m twinkle.server launch "$@"
        ;;
    *)
        printf 'Usage: twinkle-runtime {h20-a|h20-b|h20-c|head|ray-head|ray-worker|server} [options]\n' >&2
        exit 2
        ;;
esac
