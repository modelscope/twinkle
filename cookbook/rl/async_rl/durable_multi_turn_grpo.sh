#!/bin/sh
set -eu

# End-to-end GSM8K multi-turn GRPO with automatic group/batch recovery.
# Start with two devices for a smoke run:
#   sh durable_multi_turn_grpo.sh \
#       --model-gpus 1 --sampler-gpus 1 --max-steps 2
#
# The same TWINKLE_DURABLE_RUN_ID and TWINKLE_RESUME_ROOT resume the run.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TWINKLE_DURABLE_RUN_ID="${TWINKLE_DURABLE_RUN_ID:-gsm8k-durable-grpo}"
export TWINKLE_RESUME_ROOT="${TWINKLE_RESUME_ROOT:-$REPO_ROOT/output/durable-gsm8k}"

python "$SCRIPT_DIR/durable_multi_turn_grpo.py" \
    --model-id ms://Qwen/Qwen3.5-4B \
    --strategy native_fsdp \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 4 \
    --max-tokens 1024 \
    --batch-size 2 \
    --mini-batch-size 8 \
    --micro-batch-size 2 \
    --max-steps 100 \
    --lr 1e-5 \
    --lora-r 8 \
    --adapter-name durable-grpo \
    "$@"
