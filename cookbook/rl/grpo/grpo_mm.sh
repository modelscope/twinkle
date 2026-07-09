#!/bin/sh
set -eu

# GRPO Multimodal training on OlympiadBench via Ray.
# Supports multimodal math/physics problems (Chinese CEE).
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh grpo_mm.sh --model-id ms://Qwen/Qwen3.5-4B --max-steps 500

python grpo_mm.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 8 \
    --max-tokens 4096 \
    --batch-size 4 \
    --mini-batch-size 4 \
    --micro-batch-size 1 \
    --max-steps 1000 \
    --lr 1e-5 \
    --save-steps 50 \
    --adapter-name default \
    "$@"
