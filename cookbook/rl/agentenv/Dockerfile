# Sandbox template for cookbook/rl/agentenv (code-writing RL on MBPP).
#
# A general-purpose Python code interpreter: the model writes functions, runs
# them, and the training loop replays hidden unit tests inside this sandbox.
# Unlike OpenEnv's AST-interpreter backend, this is a real CPython process in a
# Firecracker microVM, so the full language (assert / try / subprocess / files)
# and arbitrary pip packages are available.
#
# Build this template ONCE against your AgentENV deployment (offline step,
# requires the aenv CLI authenticated via `aenv auth`):
#
#   aenv build cookbook/rl/agentenv/Dockerfile -t twinkle-code \
#       --cpu-count 1 --memory-mb 1024
#   aenv template watch <template-id>   # wait until the build is ready
#
# aenv build understands FROM / RUN / ENV / WORKDIR / USER instructions and
# bakes the result into a snapshot, so sandboxes boot in ~50ms with all
# dependencies pre-installed. Rebuild only when this file changes.
FROM python:3.11-slim

# Use a pip mirror if your sandbox network requires it, e.g.:
# ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir numpy sympy

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
