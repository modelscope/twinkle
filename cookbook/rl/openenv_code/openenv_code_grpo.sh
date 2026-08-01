#!/bin/sh
set -eu

# Multi-turn GRPO on MBPP with a remote OpenEnv code-interpreter server.
#
# The model writes a Python function, tests it in a remote interpreter session,
# and submits it; reward is the hidden unit-test pass rate. The environment runs
# as an ordinary HTTP/WebSocket service — no Docker, no KVM, no GPU on that side.
#
# One-time setup:
#   1. On the environment host:
#        pip install openenv
#        pip install -e /path/to/OpenEnv/envs/coding_env
#        sh serve.sh                      # note the URL it serves on
#   2. On the training host:
#        pip install openenv
#
# Capacity check: the server must host BATCH_SIZE x NUM_GENERATIONS concurrent
# sessions (32 with the defaults below). serve.sh gives WORKERS x
# MAX_CONCURRENT_ENVS = 256 by default, so there is headroom.
#
# Run (override anything at invocation):
#   OPENENV_BASE_URL=http://10.0.0.5:8000 sh openenv_code_grpo.sh --max-steps 500

# ---- OpenEnv server (a load-balancer address works here too) ----
export OPENENV_BASE_URL="${OPENENV_BASE_URL:-http://127.0.0.1:8000}"
# Environment package providing the client + Action classes.
export OPENENV_ENV_NAME="${OPENENV_ENV_NAME:-coding_env}"
# Concurrent connect/reset/score calls from the driver.
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-16}"
# Max tool-calling turns per episode.
export MAX_TURNS="${MAX_TURNS:-6}"

python openenv_code_grpo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --model-gpus 4 \
    --sampler-gpus 4 \
    --num-generations 8 \
    --max-tokens 2048 \
    --batch-size 4 \
    --mini-batch-size 8 \
    --micro-batch-size 2 \
    --max-steps 1000 \
    --lr 1e-5 \
    --lora-r 16 \
    --save-steps 500 \
    --adapter-name default \
    "$@"
