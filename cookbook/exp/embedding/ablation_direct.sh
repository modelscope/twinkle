#!/bin/bash
# Ablation 1: Direct (no RAG, no condenser)
# GPUs: 4 (gen only)
# Baseline — model solves problems without any retrieved context.

set -euo pipefail

python cookbook/exp/embedding/eval_gpqa_rag.py \
  --mode direct \
  --n 200 \
  --seed 42 \
  --output ./output/thinking_rag/ablation_direct.jsonl
