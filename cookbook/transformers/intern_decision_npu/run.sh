#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace:${PYTHONPATH:-}
torchrun --nproc_per_node=4 --master_port=29641 -m twinkle_adapter.train "$@"
