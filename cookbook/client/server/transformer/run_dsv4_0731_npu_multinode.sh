#!/usr/bin/env bash
set -euo pipefail

# Start a two-node Ascend A3 Ray cluster and launch the 32-NPU Twinkle server.
#
# Head node (starts Ray, waits for the worker, then launches Twinkle Server):
#   HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 NETWORK_IFACE=eth0 \
#     bash run_dsv4_0731_npu_multinode.sh head
#
# Worker node (joins Ray and exits, leaving the Ray daemon running):
#   HEAD_IP=10.0.0.10 NODE_IP=10.0.0.11 NETWORK_IFACE=eth0 \
#     bash run_dsv4_0731_npu_multinode.sh worker
#
# Set RESET_RAY=1 when intentionally replacing an existing local Ray runtime.

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
    echo "Usage: HEAD_IP=<head-ip> NODE_IP=<this-node-ip> NETWORK_IFACE=<iface> $0 {head|worker}" >&2
    exit 2
fi

: "${HEAD_IP:?Set HEAD_IP to the Ray head node IP}"
: "${NODE_IP:?Set NODE_IP to this node IP}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_PATH="$SCRIPT_DIR/server_config_dsv4_0731_npu_multinode.yaml"
export DSV4_MODEL_ID="${DSV4_MODEL_ID:-hf://deepseek-ai/DeepSeek-V4-Flash-0731}"

NPU_PER_NODE=16
NNODES=2
TOTAL_NPUS=$((NPU_PER_NODE * NNODES))
RAY_PORT="${RAY_PORT:-6379}"
RAY_CPUS_PER_NODE="${TWINKLE_RAY_CPUS:-$(nproc)}"
CLUSTER_WAIT_SECONDS="${CLUSTER_WAIT_SECONDS:-1800}"
NETWORK_IFACE="${NETWORK_IFACE:-eth0}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TWINKLE_TRUST_REMOTE_CODE=1
export TWINKLE_FAIL_FAST=1
export TOKENIZERS_PARALLELISM=true
export GLOO_SOCKET_IFNAME="$NETWORK_IFACE"
export HCCL_SOCKET_IFNAME="$NETWORK_IFACE"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-0}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20000}"

if [[ "$ROLE" == "head" ]]; then
    export NODE_RANK=0
else
    export NODE_RANK=1
fi

IFS=',' read -r -a VISIBLE_NPUS <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [[ "${#VISIBLE_NPUS[@]}" -ne "$NPU_PER_NODE" ]]; then
    echo "Expected $NPU_PER_NODE visible NPUs, got ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 1
fi

test -f "$CONFIG_PATH"

if [[ "$DSV4_MODEL_ID" == hf://* || "$DSV4_MODEL_ID" == ms://* ]]; then
    echo "Model will be downloaded through the configured Hub backend: $DSV4_MODEL_ID"
elif [[ -d "$DSV4_MODEL_ID" ]]; then
    test -f "$DSV4_MODEL_ID/config.json"
    test -f "$DSV4_MODEL_ID/tokenizer.json"
else
    echo "Invalid DSV4_MODEL_ID: use an existing local directory or an explicit hf:// or ms:// ID." >&2
    exit 1
fi

python3 - "$NPU_PER_NODE" <<'PY'
import sys
import torch
import torch_npu  # noqa: F401

expected = int(sys.argv[1])
if not torch.npu.is_available():
    raise SystemExit('torch.npu.is_available() is False')
actual = torch.npu.device_count()
if actual < expected:
    raise SystemExit(f'Expected at least {expected} visible NPUs, got {actual}')
print(f'Ascend check passed: visible NPU count={actual}')
PY

if [[ "${RESET_RAY:-0}" == "1" ]]; then
    ray stop --force || true
fi

if ray status >/dev/null 2>&1; then
    echo "Ray is already running on this node; set RESET_RAY=1 to replace it."
elif [[ "$ROLE" == "head" ]]; then
    ray start \
        --head \
        --node-ip-address="$HEAD_IP" \
        --port="$RAY_PORT" \
        --num-cpus="$RAY_CPUS_PER_NODE" \
        --resources="{\"NPU\": $NPU_PER_NODE}" \
        --disable-usage-stats \
        --include-dashboard=false
else
    ray start \
        --address="$HEAD_IP:$RAY_PORT" \
        --node-ip-address="$NODE_IP" \
        --num-cpus="$RAY_CPUS_PER_NODE" \
        --resources="{\"NPU\": $NPU_PER_NODE}" \
        --disable-usage-stats
fi

if [[ "$ROLE" == "worker" ]]; then
    echo "Worker joined Ray at $HEAD_IP:$RAY_PORT with $NPU_PER_NODE NPU resources."
    exit 0
fi

python3 - "$TOTAL_NPUS" "$NNODES" "$NPU_PER_NODE" "$CLUSTER_WAIT_SECONDS" <<'PY'
import sys
import time
import ray

expected_total = int(sys.argv[1])
expected_nodes = int(sys.argv[2])
expected_per_node = int(sys.argv[3])
timeout = int(sys.argv[4])
deadline = time.monotonic() + timeout

ray.init(address='auto', logging_level='ERROR')
try:
    while True:
        alive_nodes = [node for node in ray.nodes() if node.get('Alive', True)]
        npu_nodes = [node for node in alive_nodes if float(node.get('Resources', {}).get('NPU', 0)) >= expected_per_node]
        total = int(ray.cluster_resources().get('NPU', 0))
        print(f'Waiting for cluster: NPU={total}/{expected_total}, NPU nodes={len(npu_nodes)}/{expected_nodes}')
        if total >= expected_total and len(npu_nodes) >= expected_nodes:
            break
        if time.monotonic() >= deadline:
            raise SystemExit('Timed out waiting for the two-node 32-NPU Ray cluster')
        time.sleep(5)
finally:
    ray.shutdown()
PY

python3 -m twinkle.server check-config --config "$CONFIG_PATH"
echo "Launching Twinkle Server at http://$HEAD_IP:8000"
exec python3 -m twinkle.server launch --config "$CONFIG_PATH"
