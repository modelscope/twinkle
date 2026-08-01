#!/bin/sh
# Code-writing GRPO on AgentENV (one Firecracker microVM per trajectory).
#
# Prerequisites (once, offline):
#   1. Deploy AgentENV (single server, or gateway+scheduler cluster).
#      Needs /dev/kvm, kernel 6.8+, CAP_SYS_ADMIN — container instances usually
#      do not qualify; see the sandbox doc in docs.
#   2. Build the sandbox template from the Dockerfile in this folder:
#        aenv build cookbook/rl/code_rl/Dockerfile -t twinkle-code \
#            --cpu-count 1 --memory-mb 1024
#   3. On the training host: pip install e2b
#
# Memory budget: BATCH_SIZE x NUM_GENERATIONS sandboxes run at once, so the
# defaults (32) need roughly 32 x 1GB + 8GB for AgentENV and the OS.
#
# Run (override anything at invocation):
#   sh run_agentenv.sh
#   AENV_API_URL=http://10.0.0.5:8000 sh run_agentenv.sh --max-steps 500
set -eu
cd "$(dirname "$0")"

export CODE_RL_BACKEND=agentenv

# ---- AgentENV connection (the client only ever needs this ONE address) ----
# Passed through as AgentEnv(api_url=...), which internally sets the E2B SDK's
# E2B_API_URL / E2B_SANDBOX_URL (AgentENV exposes an E2B-compatible HTTP API).
# Setting E2B_API_URL alone has NO effect here — this script would silently fall
# back to the default below.
#
# SECURITY: AgentENV has no authorization. Restrict its port to this training
# host with a security group / firewall rule; never expose it to the internet.
export AENV_API_URL="${AENV_API_URL:-http://127.0.0.1:8000}"
# Template built in step 2 above.
export AENV_TEMPLATE="${AENV_TEMPLATE:-twinkle-code}"
# Sandbox lifetime; must outlast one episode plus the test replay.
export SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-600}"
# Per-command timeout inside the sandbox.
export AENV_COMMAND_TIMEOUT="${AENV_COMMAND_TIMEOUT:-60}"

. ./common_args.sh

echo "Backend: agentenv | api: $AENV_API_URL | template: $AENV_TEMPLATE"

# shellcheck disable=SC2086
python train.py $TRAIN_ARGS "$@"
