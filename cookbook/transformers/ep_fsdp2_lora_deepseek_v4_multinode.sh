#!/bin/sh
set -eu

# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# All training config passed as CLI flags. Override at invocation.

# Multi-node networking config — adjust to your cluster setup.
export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_EXEC_TIMEOUT=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=20000

NNODES=4
MASTER_ADDR=node0
MASTER_PORT=29500
NPROC_PER_NODE=16
# fsdp-size / ep-size follow the total world size.
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
# log-interval is intentionally kept in sync with gradient-accumulation-steps.
GRAD_ACCUM_STEPS=4

torchrun --nnodes=$NNODES --node_rank=${NODE_RANK:?"NODE_RANK must be set"} \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  cookbook/transformers/ep_fsdp2_lora_deepseek_v4.py \
    --model-id ms://deepseek-ai/DeepSeek-V4-Flash-bf16 \
    --dataset-id ms://swift/self-cognition \
    --template-cls DeepseekV4Template \
    --max-length 2048 \
    --fsdp-size "${WORLD_SIZE}" \
    --dp-size 1 \
    --ep-size "${WORLD_SIZE}" \
    --batch-size 64 \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
    --log-interval "${GRAD_ACCUM_STEPS}" \
    --lr 1e-4 \
    --max-grad-norm 1.0 \
    --lora-r 8 \
    --lora-alpha 32 \
    --adapter-name default \
    --output-dir ./output_dsv4_multinode \
    --scheduler-cls CosineWarmupScheduler \
    --num-warmup-steps 5 \
    --enable-ep 1 \
    "$@"

#  NODE_RANK=0 bash ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=1 bash ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=2 bash ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=3 bash ep_fsdp2_lora_deepseek_v4_multinode.sh
