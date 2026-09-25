#!/bin/sh
set -eu

# Sampling-distribution replay example.
python grpo_sampling_replay.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --strategy native_fsdp \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 8 \
    --max-tokens 4096 \
    --batch-size 8 \
    --mini-batch-size 8 \
    --micro-batch-size 2 \
    --max-steps 200 \
    --lr 1e-5 \
    --save-steps 50 \
    --adapter-name default \
    "$@"
