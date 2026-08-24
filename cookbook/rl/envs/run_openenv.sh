#!/bin/sh
set -eu
cd "$(dirname "$0")"

export CODE_RL_BACKEND=openenv

# ---- OpenEnv connection (a load-balancer address works here too) ----
# For a cross-network setup this is the LOCAL end of an SSH port forward
# (http://127.0.0.1:8000) rather than the remote address — see the deployment
# guide in docs.
export OPENENV_BASE_URL="${OPENENV_BASE_URL:-http://127.0.0.1:8000}"
export OPENENV_ENV_NAME="${OPENENV_ENV_NAME:-coding_env}"
export OPENENV_MESSAGE_TIMEOUT_S="${OPENENV_MESSAGE_TIMEOUT_S:-120}"
export MAX_TURNS="${MAX_TURNS:-6}"
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

echo "Backend: openenv | server: $OPENENV_BASE_URL | env: $OPENENV_ENV_NAME"
python train.py $TRAIN_ARGS "$@"
