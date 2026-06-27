#!/usr/bin/env bash
# Native baseline: full original context, single-turn QA, no compression, no tools.
# Compare against eval_condensed_compressed.sh on identical --dataset / --limit / --model_id.
set -euo pipefail

DATASET="${DATASET:-/mnt/data/yzhao/datasets/musique_ans_v1.0_dev.jsonl}"
MODEL_ID="${MODEL_ID:-ms://Qwen/Qwen3.5-4B}"
LIMIT="${LIMIT:-500}"
NUM_GPUS="${NUM_GPUS:-4}"
OUT_DIR="${OUT_DIR:-eval_out}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
python cookbook/exp/eval_condensed.py \
    --mode native \
    --dataset_format musique \
    --dataset "${DATASET}" \
    --model_id "${MODEL_ID}" \
    --limit "${LIMIT}" \
    --num_gpus "${NUM_GPUS}" \
    --batch_size 8 \
    --max_model_len 32768 \
    --max_new_tokens 2048 \
    --max_trajectory_tokens 8192 \
    --temperature 0.0 \
    --out_dir "${OUT_DIR}"
