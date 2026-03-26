#!/usr/bin/env bash

MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-/path/to/Megatron-LM}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export PYTHONPATH="${MEGATRON_LM_PATH}:${PYTHONPATH:-}"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
torchrun --nproc_per_node=8 cookbook/megatron/npu/tp_lora_npu.py
