#!/bin/bash
# MATH (Hendrycks) difficulty-stratified evaluation.
#
# Goal: measure how the (raw) RAG gain over direct varies with problem
# difficulty (Level 1-5). Runs raw RAG first (retrieve -> qwen3.7-max condense
# -> inject, no hint filtering; it writes the problem-id file), then direct on
# the *same* problems for a paired comparison.
#
# Usage:
#   COMPRESS_API_KEY=sk-xxx bash cookbook/exp/embedding/eval_math_by_level.sh
#
# Env knobs:
#   PER_LEVEL   problems per difficulty level (default 100 -> 500 total)
#   SEED        stratified-sampling seed (default 100; must match across runs)
#   DB_PATH     LanceDB retrieval index
#   SIM / TOPK  retrieval threshold / top-k

set -euo pipefail

export COMPRESS_API_KEY="${COMPRESS_API_KEY:?Set COMPRESS_API_KEY}"

SCRIPT="cookbook/exp/embedding/eval_gpqa_rag.py"
PER_LEVEL="${PER_LEVEL:-100}"
SEED="${SEED:-100}"
SIM="${SIM:-0.75}"
TOPK="${TOPK:-1}"
OUTDIR="./output/thinking_rag"
DB_PATH="${DB_PATH:-./output.oldemb/thinking_rag/lance.db}"

mkdir -p "$OUTDIR"

echo "============================================================"
echo " MATH by level: raw RAG (qwen3.7-max condenser, no hint)"
echo "   per_level=$PER_LEVEL seed=$SEED"
echo "============================================================"
python "$SCRIPT" \
  --dataset math --math-split test \
  --mode rag \
  --per-level "$PER_LEVEL" --seed "$SEED" \
  --db-path "$DB_PATH" \
  --sim-threshold "$SIM" --top-k "$TOPK" \
  --condense \
  --output "$OUTDIR/math_rag_results.jsonl"

echo ""
echo "============================================================"
echo " MATH by level: Direct (same problems as raw RAG)"
echo "============================================================"
# Direct reads math_rag_problem_ids.json (written above) to match the subset.
python "$SCRIPT" \
  --dataset math --math-split test \
  --mode direct \
  --per-level "$PER_LEVEL" --seed "$SEED" \
  --output "$OUTDIR/math_direct_results.jsonl"

echo ""
echo "============================================================"
echo " Done. Compare with: python cookbook/exp/embedding/compare_math_levels.py"
echo "============================================================"
