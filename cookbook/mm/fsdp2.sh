#!/bin/sh
set -eu

# Multi-modal FSDP2 + LoRA training (LaTeX OCR).
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh fsdp2.sh --model-id ms://Qwen/Qwen2.5-VL-3B-Instruct --batch-size 4

CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --nproc_per_node=2 fsdp2.py \
    --model-id ms://Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset-id ms://AI-ModelScope/LaTeX_OCR \
    --template-cls Qwen2_5VLTemplate \
    --dp-size 2 \
    --batch-size 2 \
    --lr 1e-4 \
    --gradient-accumulation-steps 4 \
    --train-samples 2000 \
    --eval-samples 100 \
    --eval-interval 200 \
    --log-interval 10 \
    "$@"
