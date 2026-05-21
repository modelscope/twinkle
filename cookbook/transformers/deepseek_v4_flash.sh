# `deepseek-ai/DeepSeek-V4-Flash` uses mixed FP4/FP8 weights.
# Convert the checkpoint before training by following:
# https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87
# Install `transformers==5.8.0` before running this cookbook.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 cookbook/transformers/deepseek_v4_flash.py
