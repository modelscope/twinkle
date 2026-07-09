#!/bin/bash
# Ablation 4: RAG + local vLLM condenser (Qwen3.5-4B-CM-v2)
# GPUs: 8 (emb=2 + gen=4 + condenser=2)
# Local 4B condenser as primary, API as fallback.

set -euo pipefail

export EVAL_CONDENSER_GPUS=2
export COMPRESS_API_KEY="${COMPRESS_API_KEY:?Set COMPRESS_API_KEY}"

python cookbook/exp/embedding/eval_gpqa_rag.py \
  --mode rag \
  --n 200 \
  --seed 42 \
  --sim-threshold 0.6 \
  --top-k 1 \
  --condense \
  --output ./output/thinking_rag/ablation_rag_local_condenser.jsonl
