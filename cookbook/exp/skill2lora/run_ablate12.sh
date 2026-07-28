#!/usr/bin/env bash
# ==============================================================================
# run_ablate12.sh — sequential launcher for the 12-experiment skill ablation.
#
# Reads the run plan from skill_ablate/config.py (single source of truth for order /
# dir names / think / skill-max-tokens / optional gate), then runs each experiment via
# `python -m skill_ablate.main --exp E{n}` in RUN_ORDER.
#
# Per experiment:
#   - isolated product dir  output.ablate12/<exp_dir>/
#   - idempotent: a successful run writes <exp_dir>/DONE.json (atomic, last step); both this
#     script and skill_ablate.main skip completed experiments unless FORCE=1
#   - env snapshot          output.ablate12/<exp_dir>/env_info.txt
#   - skill-max-tokens      8192 (think) / 4096 (nothink)  [from the plan]
#   - E12 (sft, optional)   SKIPPED unless RUN_SFT=1
#   - sleep between runs to let the previous Ray/vLLM engine tear down (avoid contention)
#
# Env knobs (all optional):
#   DEEPMATH_DIR=$HERE/../../../deepmath_103k   TRAIN_N=5000   MAX_UPDATES=50   EVAL_EVERY=5
#   LR=1e-6   RUN_SFT=1   FORCE=1   ONLY="E5 E6"   SLEEP=30   SWANLAB_PROJECT=twinkle
#   MIN_LEVEL=6   CHUNK_SIZE=32   (gradient-signal fix: E1/E5 audit — level<=5 all-pass
#   dominated, 16-problem chunks leave only ~6 mixed groups per update; eval split unaffected)
# ==============================================================================
set -euo pipefail

# avoid backward-pass OOM from allocator fragmentation (E6 crash: 15GiB reserved-unallocated);
# inherited by the Ray training actors via twinkle's runtime env passthrough
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# central env file (optional): put all knobs in one place. ENV_FILE=xxx overrides the path.
ENV_FILE="${ENV_FILE:-$HERE/ablate12.env}"
if [ -f "$ENV_FILE" ]; then
    echo "[ablate12] loading env from $ENV_FILE"
    set -a; . "$ENV_FILE"; set +a
fi

OUT_ROOT="${OUT_ROOT:-$HERE/output.ablate12}"
# DeepMath-103K (difficulty-stratified loader in skill_ablate/data.py); replaces the old
# SEAM/aops input — see skill_quality_analysis.md 组成漂移修正.
DEEPMATH_DIR="${DEEPMATH_DIR:-$(cd "$HERE/../../.." && pwd)/deepmath_103k}"
TRAIN_N="${TRAIN_N:-5000}"
EVAL_SIZE="${EVAL_SIZE:-128}"
MAX_UPDATES="${MAX_UPDATES:-50}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LR="${LR:-1e-6}"
MIN_LEVEL="${MIN_LEVEL:-6}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
SLEEP="${SLEEP:-30}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-twinkle}"
RUN_SFT="${RUN_SFT:-0}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"

mkdir -p "$OUT_ROOT"

# --- pull the run plan (name \t exp_dir \t think \t smt \t optional) -------------------
PLAN="$(python3 skill_ablate/config.py --plan)"

snapshot_env() {  # $1 = target file
    {
        echo "=== ablate12 env snapshot @ $(date -u +%FT%TZ) ==="
        echo "host: $(hostname)"
        echo "python: $(python3 -c 'import sys;print(sys.version.split()[0])')"
        echo "torch: $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo NA)"
        echo "vllm: $(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo NA)"
        echo "transformers: $(python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo NA)"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
        echo "nvidia-smi:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi NA)"
        echo "GPU layout: TRAIN=${TRAIN_GPUS:-2} REF=${REF_GPUS:-2} SKILL_SAMPLER=${SKILL_SAMPLER_GPUS:-2} BASE_SAMPLER=${BASE_SAMPLER_GPUS:-2}"
        echo "LLM_BACKUP set: $([ -n "${LLM_BACKUP_API_KEY:-}${LLM_BACKUP_BASE_URL:-}${OPENAI_API_KEY:-}" ] && echo yes || echo no)"
    } > "$1"
}

echo "[ablate12] run order:"; echo "$PLAN" | awk -F'\t' '{printf "  %s -> %s (think=%s smt=%s opt=%s)\n",$1,$2,$3,$4,$5}'

while IFS=$'\t' read -r NAME EXP_DIR THINK SMT OPTIONAL; do
    [ -z "$NAME" ] && continue
    if [ -n "$ONLY" ] && ! grep -qw "$NAME" <<< "$ONLY"; then
        echo "[ablate12] $NAME skipped (not in ONLY='$ONLY')"; continue
    fi
    if [ "$OPTIONAL" = "1" ] && [ "$RUN_SFT" != "1" ]; then
        echo "[ablate12] $NAME ($EXP_DIR) skipped: optional; set RUN_SFT=1 to run"; continue
    fi

    EXP_OUT="$OUT_ROOT/$EXP_DIR"
    if [ -f "$EXP_OUT/DONE.json" ] && [ "$FORCE" != "1" ]; then
        echo "[ablate12] $NAME already done ($EXP_OUT/DONE.json); FORCE=1 to rerun"; continue
    fi
    mkdir -p "$EXP_OUT"
    snapshot_env "$EXP_OUT/env_info.txt"

    echo "======================================================================"
    echo "[ablate12] START $NAME -> $EXP_OUT  (think=$THINK skill_max_tokens=$SMT)"
    echo "======================================================================"
    LOG="$EXP_OUT/run.log"
    FORCE_FLAG=""
    [ "$FORCE" = "1" ] && FORCE_FLAG="--force"
    set +e
    python -m skill_ablate.main \
        --exp "$NAME" \
        --deepmath-dir "$DEEPMATH_DIR" \
        --n "$TRAIN_N" \
        --eval-size "$EVAL_SIZE" \
        --output-dir "$EXP_OUT" \
        --skill-max-tokens "$SMT" \
        --max-updates "$MAX_UPDATES" \
        --eval-every-updates "$EVAL_EVERY" \
        --min-level "$MIN_LEVEL" \
        --chunk-size "$CHUNK_SIZE" \
        --lr "$LR" \
        --swanlab-project "$SWANLAB_PROJECT" \
        $FORCE_FLAG \
        2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    if [ "$RC" != "0" ]; then
        echo "[ablate12] $NAME FAILED (rc=$RC); see $LOG. Stopping."; exit "$RC"
    fi
    echo "[ablate12] $NAME done. Sleeping ${SLEEP}s for engine teardown..."
    sleep "$SLEEP"
done <<< "$PLAN"

echo "[ablate12] all requested experiments finished."
