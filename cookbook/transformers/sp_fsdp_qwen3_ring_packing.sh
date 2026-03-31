#!/bin/bash
# Packed-input sequence-parallel / ring-attention cookbook over 4 GPUs.
# This script uses Qwen3-0.6B with FlashAttention2 and a PackingDataset.
#
# If you are forcing derived ring in `_derive_sequence_parallel_sizes`, keep the
# dataset packed and preserve the effective local batch_size == 1 contract.
#
# Optional environment variables:
#   MODEL_ID=ms://Qwen/Qwen3-0.6B
#   DATASETS=ms://swift/self-cognition
#   MAX_LENGTH=2048
#   TRAIN_SLICE=500
#   EVAL_SLICE=100
#   TRAIN_BATCH_SIZE=2
#   EVAL_BATCH_SIZE=2
#   PACKING_NUM_PROC=1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 sp_fsdp_qwen3_ring_packing.py
