#!/bin/sh
set -eu

# GRPO on GSM8K with DP workload balancing across the model DP ranks.
# Thinking completions vary a lot in length; --balance-batch evens out the work per rank before each
# optimizer step. Defaults: Qwen3.5-4B, FSDP2 on 2 devices (2 data ranks) plus 2 vLLM devices, padded
# inputs, dynamic batching with at most 16 rows and 8192 tokens per micro-batch.
# Override at invocation, e.g.:
#   sh dp_balance_grpo.sh --padding-free            # packed inputs
#   sh dp_balance_grpo.sh --no-balance-batch        # baseline without balancing
#   sh dp_balance_grpo.sh --model-id /path/to/Qwen3.5-4B --dataset-id /path/to/gsm8k/train.parquet

python dp_balance_grpo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --strategy native_fsdp \
    --model-gpus 2 \
    --sampler-gpus 2 \
    --num-generations 8 \
    --max-tokens 2048 \
    --batch-size 8 \
    --mini-batch-size 32 \
    --micro-batch-size 16 \
    --max-tokens-per-micro-batch 8192 \
    --max-steps 200 \
    --lr 1e-5 \
    --lora-r 16 \
    --balance-batch \
    "$@"
