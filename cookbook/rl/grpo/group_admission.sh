#!/bin/sh
set -eu

# End-to-end GSM8K GRPO with complete-group admission and bounded resampling.
# Keep max-steps small for a smoke test, for example:
#   CUDA_VISIBLE_DEVICES=0,1 sh group_admission.sh \
#       --model-gpus 1 --sampler-gpus 1 --max-steps 2

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python "$SCRIPT_DIR/group_admission.py" \
    --model-id ms://Qwen/Qwen3.5-4B \
    --strategy native_fsdp \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 8 \
    --max-tokens 4096 \
    --batch-size 8 \
    --mini-batch-size 8 \
    --micro-batch-size 2 \
    --max-steps 1000 \
    --lr 1e-5 \
    --lora-r 16 \
    --save-steps 1000 \
    --adapter-name default \
    "$@"
