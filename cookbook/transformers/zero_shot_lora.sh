#!/bin/sh
# Generate the allocation from pretrained spectra, then train the selected FFT/LoRA modules.
# Reuse an offline allocation with: --zero-shot-config ./output/zero_shot/hybrid_config.json

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 zero_shot_lora.py \
    --model-id ms://Qwen/Qwen3.5-9B \
    --dataset-id data/financial_sft/processed/finqa_tatqa_train_messages.jsonl \
    --template-cls Qwen3_5Template \
    --fsdp-size 4 \
    --dp-size 1 \
    --batch-size 16 \
    --optimizer-cls AdamW \
    --lr 2.5e-5 \
    --weight-decay 0.01 \
    --gradient-accumulation-steps 4 \
    --log-interval 1 \
    --output-dir ./output/zero_shot_lora \
    --adapter-name default \
    --lora-r 64 \
    --scheduler-cls CosineWarmupScheduler \
    --num-warmup-steps 10 \
    --train-samples 1000 \
    --zero-shot-r 64 \
    --zero-shot-alpha 128 \
    --zero-shot-fft-ratio 0.3 \
    --zero-shot-epsilon 1e-12 \
    --zero-shot-lr-fft 1e-6 \
    --zero-shot-lr-lora 2.5e-5 \
    "$@"
