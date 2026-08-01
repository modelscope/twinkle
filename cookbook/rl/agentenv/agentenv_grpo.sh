#!/bin/sh
set -eu

# Multi-turn GRPO on MBPP with AgentENV Firecracker sandboxes.
#
# The model writes a Python function, tests it inside a real microVM sandbox and
# submits it; reward is the hidden unit-test pass rate. Sandbox placement / load
# balancing / lifecycle are handled by AgentENV; this script only needs the API
# endpoint.
#
# One-time setup:
#   1. Deploy AgentENV and note its endpoint (server :8000, or gateway :8080).
#   2. Build the sandbox template from the Dockerfile in this folder:
#        aenv auth        # point at your deployment, any non-empty API key
#        aenv build cookbook/rl/agentenv/Dockerfile -t twinkle-code \
#            --cpu-count 1 --memory-mb 1024
#        aenv template watch <template-id>
#   3. pip install e2b
#
# Run (override anything at invocation):
#   AENV_API_URL=http://10.0.0.5:8080 sh agentenv_grpo.sh --max-steps 500

# ---- AgentENV connection (client only ever needs this ONE address) ----
# NOTE: this is THIS SCRIPT's variable name. It is passed as AgentEnv(api_url=...),
# which internally sets the E2B SDK's E2B_API_URL / E2B_SANDBOX_URL (AgentENV
# exposes an E2B-compatible HTTP API). Setting E2B_API_URL alone has NO effect
# here — this script would silently fall back to the default below.
#
# SECURITY: AgentENV has no authorization. Restrict its port to this training
# host with a security group / firewall rule; never expose it to the internet.
export AENV_API_URL="${AENV_API_URL:-http://127.0.0.1:8000}"
export AENV_TEMPLATE="${AENV_TEMPLATE:-twinkle-code}"
# Sandbox idle timeout in seconds; must outlast the slowest rollout in a batch.
export SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-600}"
# Concurrent sandbox create/kill HTTP calls from the driver.
export ENV_CONCURRENCY="${ENV_CONCURRENCY:-16}"
# Max tool-calling turns per episode.
export MAX_TURNS="${MAX_TURNS:-6}"

python agentenv_grpo.py \
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
