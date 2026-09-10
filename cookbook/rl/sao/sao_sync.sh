#!/bin/sh
set -eu

# Synchronous SAO correctness baseline: 4 policy + 4 critic + 4 sampler GPUs.
# This is deliberately not the asynchronous rollout/learner pipeline.
python sao_sync.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --model-gpus 4 \
    --critic-model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 1 \
    --max-tokens 1024 \
    --batch-size 4 \
    --mini-batch-size 4 \
    --micro-batch-size 1 \
    --gamma 1.0 \
    --sao-alpha 1.5 \
    --sao-critic-lambda 1.0 \
    --critic-updates-per-actor-update 2 \
    --epsilon-low 0.3 \
    --epsilon-high 5.0 \
    --detach-importance-weight \
    --freeze-critic-attention \
    --lr 1e-6 \
    --critic-learning-rate 5e-6 \
    --max-steps 200 \
    --save-steps 50 \
    --adapter-name default \
    "$@"
