#!/bin/sh
set -eu

# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# Install `transformers==5.8.0` before running this cookbook.
# All training config passed as CLI flags. Override at invocation.

# Multi-node networking config — adjust to your cluster setup.
export GLOO_SOCKET_IFNAME="eth0"
export HCCL_SOCKET_IFNAME="eth0"
export HCCL_EXEC_TIMEOUT=1200
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_IF_BASE_PORT=20000

NNODES=4
MASTER_ADDR=node0
MASTER_PORT=29500
NPROC_PER_NODE=16

torchrun --nnodes=$NNODES --node_rank=${NODE_RANK:?"NODE_RANK must be set"} \
  --nproc_per_node=$NPROC_PER_NODE \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  ep_fsdp2_lora_deepseek_v4.py \
    --model-id ms://deepseek-ai/DeepSeek-V4-Flash-bf16 \
    --dataset-id ms://swift/self-cognition \
    --fsdp-size $NNODES*$NPROC_PER_NODE \
    --dp-size 1 \
    --ep-size $NNODES*$NPROC_PER_NODE \
    --batch-size 64 \
    --gradient-accumulation-steps 4 \
    --output-dir ./output_dsv4_multinode \
    --enable-ep 1 \
    "$@"

#  NODE_RANK=0 sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=1 sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=2 sh ep_fsdp2_lora_deepseek_v4_multinode.sh
#  NODE_RANK=3 sh ep_fsdp2_lora_deepseek_v4_multinode.sh
