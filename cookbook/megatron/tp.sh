#!/bin/sh
set -eu

# Megatron TP + LoRA training.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh tp.sh --model-id ms://Qwen/Qwen3.5-4B --tp-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 tp.py \
    --model-id ms://Qwen/Qwen3-4B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Template \
    --dp-size 1 \
    --pp-size 2 \
    --cp-size 2 \
    --tp-size 2 \
    --batch-size 8 \
    --lr 1e-4 \
    --train-samples 1000 \
    --log-interval 10 \
    --eval-interval 20 \
    --output-dir ./output/megatron_tp \
    --model-name twinkle大模型 \
    --model-author ModelScope社区 \
    "$@"
