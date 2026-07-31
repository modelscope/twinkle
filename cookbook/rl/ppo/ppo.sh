#!/bin/sh
set -eu

# Standard PPO on GSM8K via Ray.
# Transformers/Accelerate-FSDP: 4 policy + 4 full-parameter critic + 4 sampler GPUs.
# Override any option after the defaults, for example:
#   sh ppo.sh --max-steps 20 --ppo-epochs 1

python ppo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --model-gpus 4 \
    --critic-model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 2 \
    --max-tokens 1024 \
    --batch-size 4 \
    --mini-batch-size 4 \
    --micro-batch-size 1 \
    --ppo-epochs 4 \
    --gamma 1.0 \
    --gae-lambda 0.95 \
    --kl-coef 0.01 \
    --value-clip 0.2 \
    --lr 1e-5 \
    --critic-learning-rate 1e-5 \
    --max-steps 200 \
    --save-steps 50 \
    --adapter-name default \
    "$@"
