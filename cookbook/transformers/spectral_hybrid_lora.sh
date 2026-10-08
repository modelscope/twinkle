#!/bin/sh
# Train from an allocation generated with --generate-allocation-only.

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 spectral_hybrid_lora.py \
    --model-id ms://Qwen/Qwen3.5-9B \
    --dataset-id ms://swift/self-cognition \
    --template-cls Qwen3_5Template \
    --fsdp-size 4 \
    --dp-size 1 \
    --batch-size 16 \
    --optimizer-cls AdamW \
    --weight-decay 0.01 \
    --gradient-accumulation-steps 4 \
    --log-interval 1 \
    --output-dir ./output/spectral_hybrid_lora \
    --spectral-config ./output/spectral_hybrid_lora/config.json \
    --adapter-name default \
    --lora-r 64 \
    --scheduler-cls CosineWarmupScheduler \
    --num-warmup-steps 10 \
    --train-samples 1000 \
    --spectral-r 64 \
    --spectral-alpha 128 \
    --spectral-fft-ratio 0.3 \
    --spectral-epsilon 1e-12 \
    --spectral-lr-fft 1e-6 \
    --spectral-lr-lora 2.5e-5 \
    "$@"
