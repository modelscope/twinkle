#!/bin/sh
# Launch the agentic RSI run, straight to training:
#   * sandboxed workspaces (one microVM per slot) from the built template;
#   * the ms-agent harness drives twinkle's own loop and generates through the
#     local sampler, so only the forward tunnel to the sandbox tools is needed --
#     no policy endpoint is bound and nothing has to route back from the sandbox;
#   * the keyword pool is off (--num-keywords 0), so the challenger proposes from
#     scratch and the run starts without first spending rollouts inventing topics;
#   * 32 concurrent jobs throughout (envs, and the sampler's in-flight cap).
#
# Usage (ENV_FILE is required -- set it every launch, no default):
#     ENV_FILE=.temp/ablate.env sh cookbook/rsi/run.sh
#     ENV_FILE=.temp/ablate.env sh cookbook/rsi/run.sh --max-steps 50   # extra flags forwarded
#
# The sandbox connection (SANDBOX_API_URL, and the ssh tunnel that backs it) lives
# in that env file, sourced rather than duplicated here.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
cd "$REPO_ROOT"

# The packages live under src/ (editable layout), so make them importable whether
# or not this interpreter has them installed. Prepended, never clobbering.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Sourced with -a so whatever the file sets -- SANDBOX_API_URL, and any
# LLM_BACKUP_* / SWANLAB_* it carries -- is exported for rsi_grpo to read. No
# default: the env file is named explicitly at each launch.
: "${ENV_FILE:?set ENV_FILE to the run's env file, e.g. ENV_FILE=.temp/ablate12.env sh cookbook/rsi/run.sh}"
[ -f "$ENV_FILE" ] || { echo "env file not found: $ENV_FILE" >&2; exit 1; }
set -a
. "$ENV_FILE"
set +a

SANDBOX_TEMPLATE="${SANDBOX_TEMPLATE:-twinkle-rsi-msagent}"
SANDBOX_API_URL="${SANDBOX_API_URL:-http://127.0.0.1:8000}"

# Fail before the model loads if the sandbox server cannot be reached: the harness
# runs the agent's tools over the forward tunnel that backs this URL. A reachable
# server that answers 401/403 (it wants AENV_API_KEY, set in the env file) still
# counts as up -- only a connection failure aborts here.
if ! curl -sS -m 5 -o /dev/null "$SANDBOX_API_URL" 2>/dev/null; then
    echo "sandbox API unreachable at $SANDBOX_API_URL" >&2
    echo "bring up the forward tunnel first (see $ENV_FILE), e.g.:" >&2
    echo "    ssh -N -L 8000:127.0.0.1:8000 root@<sandbox-host> &" >&2
    exit 1
fi

echo "[run] template=$SANDBOX_TEMPLATE api=$SANDBOX_API_URL runner=harness envs=32 keywords=off"
exec python cookbook/rsi/rsi_grpo.py \
    --num-envs 32 \
    --num-keywords 0 \
    --sandbox-template "$SANDBOX_TEMPLATE" \
    --sandbox-api-url "$SANDBOX_API_URL" \
    --agent-config cookbook/rsi/agentic/rsi_agent.yaml \
    --agent-runner harness \
    "$@"
