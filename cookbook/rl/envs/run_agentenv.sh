#!/bin/sh
set -eu
cd "$(dirname "$0")"

export CODE_RL_BACKEND=agentenv
# LOCAL AgentENV instance or:
# ssh -N -L 8000:127.0.0.1:8000 root@xx.xx.xx.xx
# if you can connect the agent env server by ssh
export AENV_API_URL="${AENV_API_URL:-http://127.0.0.1:8000}"
export AENV_TEMPLATE="${AENV_TEMPLATE:-twinkle-code}"
# Sandbox lifetime; must outlast one episode plus the test replay.
export SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-600}"
# Per-command timeout inside the sandbox.
export AENV_COMMAND_TIMEOUT="${AENV_COMMAND_TIMEOUT:-60}"
# Max tool-calling turns per episode.
export MAX_TURNS="${MAX_TURNS:-6}"
# Concurrent reset/score calls issued from the driver.
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-16}"

TRAIN_ARGS="
    --model-id ms://Qwen/Qwen3.5-4B
    --model-gpus 4
    --sampler-gpus 4
    --num-generations 8
    --max-tokens 2048
    --batch-size 4
    --mini-batch-size 8
    --micro-batch-size 2
    --max-steps 1000
    --lr 1e-5
    --lora-r 16
    --save-steps 500
    --adapter-name default
"

echo "Backend: agentenv | api: $AENV_API_URL | template: $AENV_TEMPLATE"
python train.py $TRAIN_ARGS "$@"
