#!/bin/sh
set -eu

python short_math_grpo_routing_replay.py \
    --model-id ms://Qwen/Qwen3.6-35B-A3B \
    --strategy native_fsdp \
    --router_replay_mode R3 \
    --model-gpus 4 \
    --sampler-gpus 2 \
    --fsdp-size 4 \
    --dp-size 1 \
    --ep-size 4 \
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
