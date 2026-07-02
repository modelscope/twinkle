#!/bin/sh
set -eu

# EP + FSDP2 + LoRA training for Qwen3.5-MoE.
# Override any flag at invocation, e.g.:
#   sh ep_fsdp2_lora_qwen3_5_moe.sh --batch-size 8 --lr 5e-5

NPROC_PER_NODE=8
GRAD_ACCUM_STEPS=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc-per-node="${NPROC_PER_NODE}" \
  cookbook/transformers/ep_fsdp2_lora_qwen3_5_moe.py \
    --model-id ms://Qwen/Qwen3.6-35B-A3B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --max-length 8192 \
    --fsdp-size "${NPROC_PER_NODE}" \
    --dp-size 1 \
    --ep-size "${NPROC_PER_NODE}" \
    --batch-size 8 \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --log-interval "${GRAD_ACCUM_STEPS}" \
    --lr 1e-4 \
    --max-grad-norm 1.0 \
    --lora-r 8 \
    --lora-alpha 32 \
    --adapter-name default \
    --output-dir ./output \
    --resume-only-model 0 \
    --ignore-data-skip 0 \
    --scheduler-cls CosineWarmupScheduler \
    --num-warmup-steps 0 \
    --enable-ep 1 \
    "$@"
