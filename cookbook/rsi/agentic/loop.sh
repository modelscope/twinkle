#!/bin/bash
# Self-evolving loop: collect, train on what was collected, collect again from the
# weights that came out.
#
#   collect:  challenge.py --model-id <last ckpt> --keep-groups 8
#             runs until 8 groups have been kept, whatever that costs in topics
#   train:    train.py --run-dir <that collection>  ->  one HF checkpoint
#   repeat
#
# Nothing is generated in the training stage and nothing is re-encoded: the tokens
# trained on are the ones the sampler produced, read straight off disk.
#
# The two stages are separate processes so each gets every GPU. Collection is the
# slow half, and splitting the GPUs between a trainer and a sampler in one process
# would halve it. The cost is restarting vLLM and the sandboxes each time, measured
# at 1-2 minutes against roughly 40 minutes of collecting.
#
# Nothing about the host is written down here: the repo is found from this script's
# own location, the GPU count from nvidia-smi, and the secrets have to be exported
# first -- the script stops with the name of whatever is missing rather than
# guessing a value that would fail deep inside a run.
#
#   export E2B_API_KEY=...              # sandbox host key
#   export SANDBOX_API_URL=http://...   # sandbox host address, with port
#   export LLM_BACKUP_API_KEY=...       # dashscope, for checks/statements/rubric
#   bash cookbook/rsi/agentic/loop.sh              # until killed
#   ITERATIONS=1 bash cookbook/rsi/agentic/loop.sh # one collect + one train
set -e
# Both stages are piped into tee, and without this the pipeline's status is tee's,
# which is 0 even when python died. An earlier loop crashed inside collection,
# trained on the partial collection anyway, saved a checkpoint from it and marked
# the iteration finished -- all reported as success.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
if [ ! -f "$HERE/challenge.py" ] || [ ! -f "$REPO/setup.cfg" ]; then
    echo "expected challenge.py beside this script and the repo root three levels up" >&2
    exit 1
fi
cd "$REPO"

missing=""
for v in E2B_API_KEY SANDBOX_API_URL LLM_BACKUP_API_KEY; do
    [ -z "${!v}" ] && missing="$missing $v"
done
if [ -n "$missing" ]; then
    echo "export these first:$missing" >&2
    exit 1
fi

# Every GPU on the box unless told otherwise. Counted rather than written down,
# since the point of moving hosts is usually a different number of them.
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
[ "$GPUS" -lt 1 ] && { echo "nvidia-smi reports no GPUs" >&2; exit 1; }
DEVICES="${DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

# Refuse to start on top of someone else's job. Both stages want the whole GPU:
# challenge.py boots one vLLM per GPU at 0.8 of its memory, so sharing means an
# out-of-memory crash partway in and the other job may go down with it. Any compute
# process at all counts; CONFIRM_GPUS=1 starts anyway.
BUSY="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l)"
if [ "$BUSY" -gt 0 ] && [ "${CONFIRM_GPUS:-0}" != "1" ]; then
    echo "$BUSY process(es) already on the GPUs:" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >&2
    echo "set CONFIRM_GPUS=1 to start anyway" >&2
    exit 1
fi

export AENV_API_URL="$SANDBOX_API_URL"
export AENV_API_KEY="$E2B_API_KEY"
# Sandbox image name: a fact about how the host was set up, not a preference.
export AENV_TEMPLATE="${AENV_TEMPLATE:-twinkle-rsi-msagent}"

# ---- what to run ---------------------------------------------------------
ITERATIONS="${ITERATIONS:-0}"          # 0 = until killed
TAG="${TAG:-fix1}"
BASE_MODEL="${BASE_MODEL:-ms://Qwen/Qwen3-4B}"

# Concurrent sandboxes, i.e. how many trajectories are in flight at once. Bounded
# by the sandbox host, not by the GPUs here, so it does not follow GPUS. 32, not
# the 96 a capacity probe once managed: holding 96 for a whole run was not
# reliable. Each slot is one thread and one microVM; vLLM sees up to this many
# single-trajectory requests at a time and batches them itself.
SANDBOX_SLOTS="${SANDBOX_SLOTS:-32}"

# Groups kept per iteration, and their shape. A group is one keyword draw answered
# GROUP_SIZE times; it is kept when at least one of those answers became a task the
# solver passes sometimes, meaning n_pass in [1, SOLVER_ROLLOUTS-1]. Outside that
# band every attempt carries the same reward, the group mean equals it, and the
# advantage is zero for all of them.
#
#   proposing side: KEEP_GROUPS x GROUP_SIZE       = 8 x 8 = 64 trajectories
#   solving side:   KEEP_GROUPS x SOLVER_ROLLOUTS  = 8 x 8 = 64 trajectories
#                   128 total, one optimizer step over all of it
#
# What this pays for and throws away: every proposal that produced a task costs
# SOLVER_ROLLOUTS sandbox attempts, and only the selected proposal's attempts are
# trained on. At 8 groups x 8 proposals that is up to 512 attempts run and 64
# trained on. The unselected proposals are not wasted on the proposing side -- each
# earns its own reward from its own n_pass, including 0 for the ones that produced
# no task at all.
KEEP_GROUPS="${KEEP_GROUPS:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SOLVER_ROLLOUTS="${SOLVER_ROLLOUTS:-8}"

# Cap on files one build may leave behind, appended to the system prompt. This
# changes the prompt and therefore what is trained; 4 is what every run since it
# was added has used.
MAX_BUILD_FILES="${MAX_BUILD_FILES:-4}"

# Reasoning cap on every API call. The one knob that moved wall-clock: 58s -> 10s
# per turn at 2048 on a ~15k-character context.
API_THINKING_BUDGET="${API_THINKING_BUDGET:-4096}"
API_MODEL="${API_MODEL:-qwen3.8-max}"
API_BASE="${API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

# Novelty. The bank is one file for the whole loop, not one per iteration, because
# the point of it is comparing iteration k+1's proposals against what k produced.
# TASK_BANK="" turns it off and gives back the pass-rate gaussian alone;
# NOVELTY_FLOOR=1 keeps the judging and the log but stops it changing any reward.
#
# 1 is the default because the score it would multiply in has not been shown to carry
# anything yet. Measured on the 27 proposals of iter1 (novelty_scores.jsonl): judged
# against their own siblings, 24 of 27 scored exactly 0.0, so the term was constant
# across the group and contributed nothing once GRPO subtracts the group mean -- while
# still halving every proposer reward at floor 0.5. The alternative measured, labelling
# each task's shape on its own and scoring by how rare that shape is in the group, does
# separate proposals (0 of 4 groups constant), but its label changed between sampled
# repeats on 10 of 27 statements, so the number it produces is not comparable across
# runs. Until one of those is fixed the score is written to novelty_scores.jsonl and
# read there. Set 0.5 to bring it back into the reward.
NOVELTY_FLOOR="${NOVELTY_FLOOR:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-6}"
SIDES="${SIDES:-both}"

ROOT="output/rsi_agentic/${TAG}"
# One checkpoint directory for the whole loop, overwritten every iteration, so the
# disk holds one 4B model rather than one per iteration. The previous round's
# weights are gone once the next save starts: if a save dies partway there is
# nothing to fall back to but BASE_MODEL.
#
# Overridable because this is the one path whose filesystem shows up in wall-clock:
# every iteration reads it 1 + GPUS times, once per vLLM worker at collect and once
# more at train, so 7.6 GB of weights is around 60 GB of reads per iteration. Two
# filesystems on this host, same 1.5 T free, measured with dd at 1.5 GB: the repo's
# own disk reads at 223 MB/s and the parallel one at 1074 MB/s -- 4.8x, which showed
# up as five minutes of vLLM startup before any topic was launched. Left at $ROOT/ckpt
# by default so a host with one disk needs to know nothing about this.
CKPT_DIR="${CKPT_DIR:-$ROOT/ckpt}"
# Written with ${VAR-default} rather than ${VAR:-default} so that TASK_BANK=""
# means off; with the colon an empty value would silently get the default back.
TASK_BANK="${TASK_BANK-$ROOT/task_bank.jsonl}"
mkdir -p "$ROOT"

# Pick up where a previous invocation left off. The iteration number comes from a
# marker written after the checkpoint has been checked, not from train_summary.json,
# which is written at the end of a training run but would still be there after a
# crash in a later stage.
MODEL="$BASE_MODEL"
START=1
while [ -f "$ROOT/iter${START}/iteration.done" ]; do
    START=$((START + 1))
done
if [ "$START" -gt 1 ]; then
    if [ -f "$CKPT_DIR/model/config.json" ]; then
        MODEL="$CKPT_DIR/model"
    else
        echo "$((START - 1)) iteration(s) finished under $ROOT but no checkpoint at" >&2
        echo "$CKPT_DIR/model -- each iteration overwrites the one before, so those" >&2
        echo "weights are gone. Start a new TAG, or delete the iteration.done" >&2
        echo "markers to redo them from $BASE_MODEL." >&2
        exit 1
    fi
fi

cat <<EOF
=== repo         $REPO
=== gpus         $GPUS (devices $DEVICES)
=== sandbox      $AENV_API_URL template $AENV_TEMPLATE slots $SANDBOX_SLOTS
=== api          $API_MODEL at $API_BASE, thinking budget $API_THINKING_BUDGET
=== per iter     $KEEP_GROUPS groups of $GROUP_SIZE, band [1, $((SOLVER_ROLLOUTS - 1))] of $SOLVER_ROLLOUTS
=== build cap    $([ "$MAX_BUILD_FILES" -eq 0 ] && echo "none" || echo "$MAX_BUILD_FILES files, in the system prompt")
=== novelty      $([ -z "$TASK_BANK" ] && echo "off" || echo "bank $TASK_BANK, floor $NOVELTY_FLOOR")
=== trains on    $((KEEP_GROUPS * GROUP_SIZE)) propose + $((KEEP_GROUPS * SOLVER_ROLLOUTS)) solve trajectories, one step, lr $LEARNING_RATE
=== checkpoint   $CKPT_DIR/model, overwritten each iteration
=== iterations   $([ "$ITERATIONS" -eq 0 ] && echo "until killed" || echo "$ITERATIONS")
=== swanlab      ${RSI_SWANLAB_MODE:-online} project ${RSI_SWANLAB_PROJECT:-twinkle-rsi-agentic}, experiment $TAG, one step per iteration
=== starting at  iteration $START from $MODEL
EOF

i="$START"
while [ "$ITERATIONS" -eq 0 ] || [ "$i" -lt $((START + ITERATIONS)) ]; do
    OUT="$ROOT/iter${i}"
    mkdir -p "$OUT"
    echo "=== iteration $i: collect $KEEP_GROUPS groups from $MODEL -> $OUT"

    CUDA_VISIBLE_DEVICES="$DEVICES" python cookbook/rsi/agentic/challenge.py \
        --model-id "$MODEL" \
        --sampler-gpus "$GPUS" \
        --sandbox-slots "$SANDBOX_SLOTS" \
        --keep-groups "$KEEP_GROUPS" \
        --group-size "$GROUP_SIZE" \
        --solver-rollouts "$SOLVER_ROLLOUTS" \
        --max-build-files "$MAX_BUILD_FILES" \
        --api-model "$API_MODEL" \
        --api-base "$API_BASE" \
        --api-thinking-budget "$API_THINKING_BUDGET" \
        --task-bank "$TASK_BANK" \
        --novelty-floor "$NOVELTY_FLOOR" \
        --out-dir "$OUT" \
        --keyword-db "$ROOT/keywords.jsonl" \
        2>&1 | tee "$OUT/challenge.log"

    echo "=== iteration $i: train on $OUT -> $CKPT_DIR"
    # expandable_segments on the training stage only, and not on collect: one padded
    # trajectory per micro batch means every micro batch is a new shape (119 distinct
    # lengths in 128 trajectories, 7k-19k tokens), and the caching allocator cannot
    # reuse a block across sizes, so it grew to 87.8 GiB reserved against 29.0 GiB
    # live on a 97.4 GiB card. That is what starved NCCL of the few hundred MiB it
    # needs to connect the metric gather's communicator, which hung iteration 2 for
    # 54 minutes. Expandable segments let one virtual range serve every shape, so
    # reserved tracks the real peak instead of the sum of shapes. Left off for
    # collect because that stage is vLLM, which profiles its own KV cache against
    # allocator behaviour and has nothing to do with this failure.
    RSI_RUN_DIR="$OUT" \
    RSI_SAVE_DIR="$CKPT_DIR" \
    RSI_SAVE_NAME="model" \
    RSI_SIDES="$SIDES" \
    RSI_TAG="$TAG" \
    RSI_ITER="$i" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="$DEVICES" python cookbook/rsi/agentic/train.py \
        --model_id "$MODEL" \
        --model_gpus "$GPUS" \
        --lr "$LEARNING_RATE" \
        2>&1 | tee "$OUT/train.log"

    # HF-format weights plus tokenizer, which is what --model-id takes, so the next
    # iteration needs no conversion step.
    MODEL="$CKPT_DIR/model"
    if [ ! -f "$MODEL/config.json" ]; then
        echo "iteration $i saved no loadable checkpoint at $MODEL" >&2
        exit 1
    fi
    # Written last, so resuming counts only iterations whose weights are on disk.
    touch "$OUT/iteration.done"
    echo "=== iteration $i done; next starts from $MODEL"
    i=$((i + 1))
done
echo "=== stopped after iteration $((i - 1)); model at $MODEL"
