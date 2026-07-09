#!/bin/sh
set -eu

# DPO Full-Parameter Training via Ray.
# Uses separate policy and reference model GPU groups.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh dpo_full.sh --model-id ms://Qwen/Qwen3-8B --beta 0.05

python dpo_full.py \
    --model-id ms://Qwen/Qwen3-4B \
    --dataset-id ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --model-gpus 4 \
    --ref-model-gpus 4 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --lr 1e-5 \
    --beta 0.1 \
    --sft-weight 1.0 \
    --loss-type sigmoid \
    --max-length 2048 \
    --save-steps 100 \
    "$@"
