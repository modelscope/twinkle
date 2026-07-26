#!/bin/bash
# Ablation 3: RAG + API condenser (qwen3.7-max)
# GPUs: 6 (emb=2 + gen=4), condenser via API (no local vLLM)
# Compresses thinking_raw with COMPRESS_SYSTEM + CONDENSE_EVAL_QUERY via API.

set -euo pipefail

export COMPRESS_API_KEY="${COMPRESS_API_KEY:?Set COMPRESS_API_KEY}"

python cookbook/exp/embedding/eval_gpqa_rag.py \
  --mode rag \
  --n 200 \
  --seed 42 \
  --sim-threshold 0.6 \
  --top-k 1 \
  --condense \
  --output ./output/thinking_rag/ablation_rag_api_condenser.jsonl
