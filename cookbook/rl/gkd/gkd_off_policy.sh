#!/bin/sh
set -eu

# GKD Off-Policy Distillation via Ray.
# Teacher vLLM computes prompt logprobs on existing dataset responses.
# Student Megatron model learns to match teacher's token distribution.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh gkd_off_policy.sh --student-model-id ms://Qwen/Qwen3-1.7B --gkd-beta 0.3

python gkd_off_policy.py \
    --student-model-id ms://Qwen/Qwen3-0.6B \
    --teacher-model-id ms://Qwen/Qwen3-8B \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --batch-size 16 \
    --max-steps 1000 \
    --lr 5e-5 \
    --gkd-beta 0.5 \
    --gkd-temperature 1.0 \
    --gkd-topk 64 \
    --adapter-name default \
    "$@"
