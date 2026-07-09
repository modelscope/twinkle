#!/bin/sh
set -eu

# GRPO training on GSM8K via Ray.
# Model + vLLM sampler on separate GPU groups.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh grpo.sh --model-id ms://Qwen/Qwen3.5-4B --max-steps 500

python grpo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
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
