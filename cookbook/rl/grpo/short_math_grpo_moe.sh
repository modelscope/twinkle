#!/bin/sh
set -eu

# GRPO Short Math MoE on GSM8K via Ray.
# Uses Megatron MoE model with TP+EP+PP parallelism.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh short_math_grpo_moe.sh --model-id ms://Qwen/Qwen3.6-35B-A3B --max-steps 500

python short_math_grpo_moe.py \
    --model-id ms://Qwen/Qwen3.6-35B-A3B \
    --model-gpus 4 \
    --sampler-gpus 2 \
    --ep-size 2 \
    --tp-size 2 \
    --pp-size 2 \
    --tensor-parallel-size 2 \
    --num-generations 8 \
    --max-tokens 4096 \
    --batch-size 4 \
    --mini-batch-size 4 \
    --micro-batch-size 1 \
    --max-steps 1000 \
    --lr 5e-5 \
    --lora-r 16 \
    --save-steps 1000 \
    --adapter-name default \
    "$@"
