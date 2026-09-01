#!/bin/bash
# Start (or continue) a run of the resident loop.
#
# Everything about what to collect and how to train lives in rsi.py's arguments;
# this only sets up the process. Extra arguments are passed straight through, so
# anything in `python cookbook/rsi/agentic/rsi.py --help` works here:
#
#     TAG=v4 bash cookbook/rsi/agentic/run.sh
#     TAG=v4 bash cookbook/rsi/agentic/run.sh --keep-groups 4 --iterations 1
#
# Restarting the same TAG continues it from the last finished iteration.
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
if [ ! -f "$HERE/rsi.py" ] || [ ! -f "$REPO/setup.cfg" ]; then
    echo "expected rsi.py beside this script and the repo root three levels up" >&2
    exit 1
fi
cd "$REPO"

missing=""
for v in TAG E2B_API_KEY SANDBOX_API_URL LLM_BACKUP_API_KEY; do
    [ -z "${!v:-}" ] && missing="$missing $v"
done
if [ -n "$missing" ]; then
    echo "set these first:$missing" >&2
    echo "  TAG names the run; the other three are the sandbox host and dashscope" >&2
    exit 1
fi

MODEL_GPUS="${MODEL_GPUS:-2}"
SAMPLER_GPUS="${SAMPLER_GPUS:-6}"
GPUS=$((MODEL_GPUS + SAMPLER_GPUS))
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

# Refuse to start on top of another job. Both halves want whole cards -- the
# sampler takes 0.8 of each of its own and the trainer holds ~40 GB of weights and
# optimizer state for the whole run -- so sharing means an out-of-memory crash
# partway in, and the other job may go down with it. This guard has already caught
# the case worth catching: a previous run's actors still exiting, each still
# holding tens of GB, at the moment a new one started. CONFIRM_GPUS=1 starts anyway.
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l)"
if [ "$BUSY" -gt 0 ] && [ "${CONFIRM_GPUS:-0}" != "1" ]; then
    echo "$BUSY process(es) already on the GPUs:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
    echo "set CONFIRM_GPUS=1 to start anyway" >&2
    exit 1
fi

# One padded trajectory per micro batch means every micro batch is a new shape, and
# the caching allocator cannot reuse a block across sizes: on v3 it grew to 87.8 GiB
# reserved against 29.0 GiB live and starved NCCL of the few hundred MB it needs to
# connect a communicator, which hung an iteration for 54 minutes. Expandable
# segments let one virtual range serve every shape.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TWINKLE_DISABLE_CUDNN_SDP="${TWINKLE_DISABLE_CUDNN_SDP:-1}"
# INFO here is tens of thousands of lines per iteration through Ray's log forwarding,
# which is worth having only while chasing a collective.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ROOT="${ROOT:-output/rsi_agentic}"
mkdir -p "$ROOT/$TAG"
LOG="$ROOT/$TAG/run.log"
echo "=== $TAG: $MODEL_GPUS trainer + $SAMPLER_GPUS sampler GPUs, logging to $LOG"

# tee rather than a redirect so a foreground run is watchable, and pipefail above
# so the exit status is python's and not tee's.
python cookbook/rsi/agentic/rsi.py \
    --tag "$TAG" \
    --model-gpus "$MODEL_GPUS" \
    --sampler-gpus "$SAMPLER_GPUS" \
    --root "$ROOT" \
    "$@" 2>&1 | tee -a "$LOG"
