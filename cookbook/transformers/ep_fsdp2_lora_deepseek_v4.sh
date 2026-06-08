#!/usr/bin/env bash
set -euo pipefail

# EP + FSDP2 + LoRA training for DeepSeek-V4.
# ENABLE_EP=1 trains expert LoRA with target_parameters.
# ENABLE_EP=0 runs plain FSDP2 LoRA and does not train expert parameters.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export ENABLE_EP="${ENABLE_EP:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_dsv4}"

torchrun --nproc-per-node="${NPROC_PER_NODE}" \
  cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py
