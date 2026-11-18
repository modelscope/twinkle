#!/usr/bin/env bash
set -eo pipefail

MODE="${1:?usage: run_gate1_a2.sh <grpo|graphcredit> [steps] [eval-tasks] [output-dir]}"
STEPS="${2:-20}"
EVAL_TASKS="${3:-24}"
OUTPUT_DIR="${4:-outputs/graphcredit-gate1/${MODE}}"

MODEL_PATH="${MODEL_PATH:-/workspace/shared_assets/models/Qwen/Qwen3-0.6B}"
SEED="${SEED:-20261118}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
DISTANCE="${DISTANCE:-2}"
GRID_SIZE="${GRID_SIZE:-5}"
EVAL_EVERY="${EVAL_EVERY:-0}"
MAX_STEP_ATTEMPTS="${MAX_STEP_ATTEMPTS:-6}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PYTHONPATH="/workspace/graphcredit-gate0/src:/workspace/graphcredit-deps${PYTHONPATH:+:${PYTHONPATH}}"

cd /workspace/graphcredit-gate0
exec /workspace/npu-ai-env/bin/python cookbook/rl/graphcredit/gate1.py \
  --model-path "${MODEL_PATH}" \
  --mode "${MODE}" \
  --steps "${STEPS}" \
  --eval-tasks "${EVAL_TASKS}" \
  --eval-every "${EVAL_EVERY}" \
  --group-size "${GROUP_SIZE}" \
  --max-step-attempts "${MAX_STEP_ATTEMPTS}" \
  --distance "${DISTANCE}" \
  --grid-size "${GRID_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --device npu:0 \
  --output-dir "${OUTPUT_DIR}"
