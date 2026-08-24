#!/bin/sh
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh fsdp2.sh --batch-size 16 --lr 5e-5
#
# Liger fused linear cross-entropy: --enable-liger turns on the fused-CE loss
# (pair with --no-fused-ce to make it a no-op). Per-layer kernels are NOT
# gated by this flag — they always come from `kernelize(model)`'s default
# config (NPU: CANN-first chains). To force Liger per-layer kernels, pass a
# custom mapping with liger-first KernelChoice chains (see Kernel.md).
#   sh fsdp2.sh --enable-liger            # enable Liger fused-CE loss
#   sh fsdp2.sh --no-enable-liger         # disable fused-CE (default)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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
