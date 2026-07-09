#!/bin/sh
set -eu

# FSDP + Sequence Parallelism training.
# To enable Transformers sequence parallelism, set ulysses-size > 1.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh sp_fsdp_dense.sh --model-id ms://Qwen/Qwen3.5-4B --ulysses-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 sp_fsdp_dense.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --dp-size 2 \
    --fsdp-size 2 \
    --ulysses-size 2 \
    --batch-size 4 \
    --lr 1e-4 \
    --gradient-accumulation-steps 2 \
    --train-samples 500 \
    --scheduler-cls CosineWarmupScheduler \
    --log-interval 10 \
    --model-name twinkle模型 \
    --model-author twinkle团队 \
    "$@"
