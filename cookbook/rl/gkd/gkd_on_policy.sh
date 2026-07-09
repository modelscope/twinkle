#!/bin/sh
set -eu

# GKD On-Policy Multimodal Distillation via Ray.
# Student generates on-policy, teacher provides top-k prompt logprobs,
# student trains to match teacher's distribution.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh gkd_on_policy.sh --student-model-id ms://Qwen/Qwen3.5-4B --max-steps 500

python gkd_on_policy.py \
    --student-model-id ms://Qwen/Qwen3.5-4B \
    --teacher-model-id ms://Qwen/Qwen3.5-9B \
    --model-gpus 4 \
    --sampler-gpus 2 \
    --batch-size 4 \
    --max-steps 1000 \
    --max-tokens 2048 \
    --lr 5e-5 \
    --num-samples 1 \
    --gkd-beta 0.5 \
    --gkd-temperature 1.0 \
    --gkd-topk 64 \
    --adapter-name default \
    "$@"
