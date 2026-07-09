#!/bin/bash
# Ablation 2: RAG + raw thinking (no condenser, truncated to max-trace-len)
# GPUs: 6 (emb=2 + gen=4)
# Uses thinking_raw directly, truncated to 4000 chars.

set -euo pipefail

python cookbook/exp/embedding/eval_gpqa_rag.py \
  --mode rag \
  --n 200 \
  --seed 42 \
  --sim-threshold 0.6 \
  --top-k 1 \
  --max-trace-len 4000 \
  --output ./output/thinking_rag/ablation_rag_raw.jsonl
