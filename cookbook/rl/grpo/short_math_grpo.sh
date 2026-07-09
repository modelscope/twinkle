#!/bin/sh
set -eu

# GRPO Short Math Reasoning on GSM8K via Ray.
# Uses short reasoning format: shorter thinking gets higher brevity reward.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh short_math_grpo.sh --model-id ms://Qwen/Qwen3.5-4B --max-steps 500

python short_math_grpo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 8 \
    --max-tokens 4096 \
    --batch-size 8 \
    --mini-batch-size 8 \
    --micro-batch-size 2 \
    --max-steps 1000 \
    --lr 1e-5 \
    --lora-r 16 \
    --save-steps 1000 \
    --adapter-name default \
    "$@"
