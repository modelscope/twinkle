#!/bin/sh
# Compressed run: chunk → condense via Qwen3.5-4B-Condenser LoRA → extract_condensed tool loop.
# Identical --dataset / --limit / --model_id as eval_condensed_native.sh for an A/B comparison.
set -eu

DATASET="/mnt/data/yzhao/datasets/musique_ans_v1.0_dev.jsonl"
MODEL_ID="ms://Qwen/Qwen3.5-4B"
CONDENSER_LORA="ms://twinkle-kit/Qwen3.5-4B-Condenser"
LIMIT="500"
NUM_GPUS="4"
OUT_DIR="eval_out"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python cookbook/exp/eval_condensed.py \
    --mode condensed \
    --dataset_format musique \
    --dataset "${DATASET}" \
    --model_id "${MODEL_ID}" \
    --condenser_lora "${CONDENSER_LORA}" \
    --limit "${LIMIT}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size 8 \
    --max_model_len 32768 \
    --max_new_tokens 2048 \
    --max_turns 4 \
    --max_trajectory_tokens 8192 \
    --chunk_size 1024 \
    --temperature 0.0 \
    --out_dir "${OUT_DIR}"
