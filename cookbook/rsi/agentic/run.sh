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
for v in TAG E2B_API_KEY SANDBOX_API_URL LLM_BACKUP_API_KEY LLM_BACKUP_MODEL \
         LLM_BACKUP_BASE_URL; do
    [ -z "${!v:-}" ] && missing="$missing $v"
done
if [ -n "$missing" ]; then
    echo "set these first:$missing" >&2
    echo "  TAG names the run; the rest are the sandbox host and the API judge" >&2
    echo "  (LLM_BACKUP_MODEL and LLM_BACKUP_BASE_URL are where --api-model and" >&2
    echo "   --api-base get their defaults, so an empty one is a run with no judge)" >&2
    exit 1
fi

# The interpreter, checked rather than hardcoded: a login shell without the conda
# environment active resolves python to /usr/local/bin/python, whose megatron.core
# has no transformer-engine metadata and raises PackageNotFoundError on import --
# after Ray is up, which reads as eight actors dying for no stated reason.
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c 'import twinkle, megatron.core' 2>/dev/null; then
    echo "$PYTHON cannot import twinkle and megatron.core: activate the environment" >&2
    echo "  twinkle was installed into, or point PYTHON at its interpreter" >&2
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
# Not under ROOT: ROOT is on the NAS, and this is 7.6 GB of weights every iteration
# and ~48 GB more on the iterations that include the optimizer.
CKPT_DIR="${CKPT_DIR:-/mnt/data2/rsi_agentic/$TAG/ckpt}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/workspace/.cache/modelscope/hub}"
# Pushed to the cloud dashboard. This works on the default project and not on the
# older twinkle-rsi-agentic one, which answers this client (0.7.17) with 422 and,
# since swanlab.init is no longer guarded, would stop the run in its first second.
# SWANLAB_MODE=local writes swanlog/ instead, for `swanlab watch`.
SWANLAB_MODE="${SWANLAB_MODE:-online}"
# The sequence limit that bounds training memory. Deliberately not given a value
# here: rsi.py's default (16384) is the measured one and duplicating the number in
# two places is how the two drift apart. Set MAX_TRAIN_LEN to override it, which is
# what a change of model, vocabulary, or trainer GPU count calls for -- the bound is
# vocab x length x 2 bytes of logits against whatever the card has left.
LIMIT=()
if [ -n "${MAX_TRAIN_LEN:-}" ]; then
    LIMIT=(--max-train-len "$MAX_TRAIN_LEN")
fi
mkdir -p "$ROOT/$TAG"
LOG="$ROOT/$TAG/run.log"
echo "=== $TAG: $MODEL_GPUS trainer + $SAMPLER_GPUS sampler GPUs, checkpoint $CKPT_DIR,"
echo "===      swanlab $SWANLAB_MODE, logging to $LOG"

# tee rather than a redirect so a foreground run is watchable, and pipefail above
# so the exit status is python's and not tee's.
#
# Every knob that decides what gets collected or how it is trained is left to
# rsi.py's own defaults on purpose. The one time a verified setting lived in a
# launcher instead -- --api-thinking-budget 4096, in a throwaway script under
# .temp -- a restart that retyped the command line dropped it, the rubric judge
# went back to thinking without a cap, 43% of its calls hit the 120s timeout and
# took their whole group down, and the iteration ran at three times its usual
# wall-clock before anyone noticed.
$PYTHON cookbook/rsi/agentic/rsi.py \
    --tag "$TAG" \
    --model-gpus "$MODEL_GPUS" \
    --sampler-gpus "$SAMPLER_GPUS" \
    --root "$ROOT" \
    --ckpt-dir "$CKPT_DIR" \
    --swanlab-mode "$SWANLAB_MODE" \
    ${LIMIT[@]+"${LIMIT[@]}"} \
    "$@" 2>&1 | tee -a "$LOG"
