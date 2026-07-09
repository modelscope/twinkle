#!/bin/sh
set -eu

# Multi-modal FSDP2 + LoRA training for Gemma4 (LaTeX OCR).
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh fsdp2_gemma4_mm.sh --model-id ms://google/gemma-4-4b-it --batch-size 4

CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --nnodes=1 --nproc_per_node=2 fsdp2_gemma4_mm.py \
    --model-id ms://google/gemma-4-12b-it \
    --dataset-id ms://AI-ModelScope/LaTeX_OCR \
    --template-cls Gemma4Template \
    --dp-size 2 \
    --batch-size 2 \
    --lr 1e-4 \
    --gradient-accumulation-steps 4 \
    --train-samples 2000 \
    --eval-samples 100 \
    --log-interval 10 \
    --save-steps 200 \
    "$@"
