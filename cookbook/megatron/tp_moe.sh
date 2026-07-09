#!/bin/sh
set -eu

# Megatron TP + MoE + LoRA training.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh tp_moe.sh --model-id ms://Qwen/Qwen3.5-30B-A3B --tp-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 tp_moe.py \
    --model-id ms://Qwen/Qwen3.5-30B-A3B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --dp-size 2 \
    --tp-size 2 \
    --pp-size 2 \
    --ep-size 2 \
    --sequence-parallel \
    --batch-size 8 \
    --lr 1e-4 \
    --train-samples 1000 \
    --log-interval 10 \
    --eval-interval 20 \
    --model-name twinkle大模型 \
    --model-author ModelScope社区 \
    "$@"
