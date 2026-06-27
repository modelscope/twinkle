#!/usr/bin/env bash
# All training config passed as CLI flags. Override at invocation, e.g.:
#   bash fsdp2.sh --batch-size 16 --lr 5e-5

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  torchrun --nproc_per_node=8 fsdp2.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --fsdp-size 2 \
    --dp-size 4 \
    --batch-size 8 \
    --lr 1e-4 \
    --gradient-accumulation-steps 2 \
    --log-interval 20 \
    --eval-interval 40 \
    --eval-samples 100 \
    --output-dir ./output/fsdp2 \
    --adapter-name default \
    --scheduler-cls CosineWarmupScheduler \
    --num-warmup-steps 5 \
    --train-samples 1000 \
    --model-name twinkle大模型 \
    --model-author ModelScope社区 \
    "$@"
