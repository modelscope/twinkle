#!/bin/sh
set -eu

# DPO MultiLoRA Training via Ray (Megatron backend).
# Uses base model (disable_lora=True) as reference model.
# All training config passed as CLI flags. Override at invocation, e.g.:
#   sh dpo_multi_lora.sh --model-id ms://Qwen/Qwen3.5-4B --lr 5e-5

python dpo_multi_lora.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --dataset-id ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --model-gpus 2 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --lr 1e-4 \
    --beta 0.1 \
    --sft-weight 1.0 \
    --loss-type sigmoid \
    --max-length 2048 \
    --save-steps 100 \
    --adapter-name default_0 \
    "$@"
