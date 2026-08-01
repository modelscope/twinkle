#!/bin/sh
# Code-writing GRPO on a remote OpenEnv server (one session per trajectory).
#
# Prerequisites:
#   1. On the environment host:
#        pip install openenv && pip install -e /path/to/OpenEnv/envs/coding_env
#        HOST=127.0.0.1 sh serve.sh          # see serve.sh for network options
#   2. On the training host:
#        pip install openenv
#
# Run (override anything at invocation):
#   sh run_openenv.sh
#   OPENENV_BASE_URL=http://10.0.0.5:8000 sh run_openenv.sh --max-steps 500
set -eu
cd "$(dirname "$0")"

export CODE_RL_BACKEND=openenv

# ---- OpenEnv connection (a load-balancer address works here too) ----
# For a cross-network setup this is the LOCAL end of an SSH port forward
# (http://127.0.0.1:8000) rather than the remote address — see the deployment
# guide in docs.
export OPENENV_BASE_URL="${OPENENV_BASE_URL:-http://127.0.0.1:8000}"
# Environment package providing the client + Action classes.
export OPENENV_ENV_NAME="${OPENENV_ENV_NAME:-coding_env}"
# Per-message timeout. The executor has its own caps on top of this.
export OPENENV_MESSAGE_TIMEOUT_S="${OPENENV_MESSAGE_TIMEOUT_S:-120}"

. ./common_args.sh

echo "Backend: openenv | server: $OPENENV_BASE_URL | env: $OPENENV_ENV_NAME"

# shellcheck disable=SC2086
python train.py $TRAIN_ARGS "$@"
