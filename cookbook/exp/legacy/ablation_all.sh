#!/bin/bash
# RAG Ablation Suite — 串行运行全部 5 个消融实验
# GPUs: 需要 8 卡（兼容所有配置的最大需求）
#
# 用法:
#   COMPRESS_API_KEY=sk-xxx bash cookbook/exp/embedding/ablation_all.sh

set -euo pipefail

export COMPRESS_API_KEY="${COMPRESS_API_KEY:?Set COMPRESS_API_KEY}"

SCRIPT="cookbook/exp/embedding/eval_gpqa_rag.py"
N=500
SEED=100
SIM=0.6
TOPK=1
OUTDIR="./output/thinking_rag"
DB_PATH="./output.oldemb/thinking_rag/lance.db"

# echo "============================================================"
# echo " Ablation 1/5: Direct (no RAG)"
# echo "============================================================"
# GEN_GPUS=8 python $SCRIPT \
#   --mode direct --n $N --seed $SEED \
#   --output $OUTDIR/ablation_direct_65k.jsonl

# echo ""
# echo "============================================================"
# echo " Ablation 2/5: RAG + raw thinking (drop >24k, no condenser)"
# echo "============================================================"
# python $SCRIPT \
#   --mode rag --n $N --seed $SEED \
#   --db-path $DB_PATH \
#   --sim-threshold $SIM --top-k $TOPK \
#   --max-trace-len 24000 \
#   --output $OUTDIR/ablation_rag_raw_24k.jsonl

echo ""
echo "============================================================"
echo " Ablation 3/5: RAG + API condenser (qwen3.7-max)"
echo "============================================================"
python $SCRIPT \
  --mode rag --n $N --seed $SEED \
  --db-path $DB_PATH \
  --sim-threshold $SIM --top-k $TOPK \
  --condense \
  --output $OUTDIR/ablation_rag_api_condenser_65k.jsonl

echo ""
echo "============================================================"
echo " Ablation 4/5: RAG + local vLLM condenser (4B) + API fallback"
echo "============================================================"
EVAL_CONDENSER_GPUS=2 python $SCRIPT \
  --mode rag --n $N --seed $SEED \
  --db-path $DB_PATH \
  --sim-threshold $SIM --top-k $TOPK \
  --condense \
  --output $OUTDIR/ablation_rag_local_condenser_65k.jsonl

# echo ""
# echo "============================================================"
# echo " Ablation 5/5: RAG + cot_compressed (pre-compressed, no runtime condenser)"
# echo "============================================================"
# python $SCRIPT \
#   --mode rag --n $N --seed $SEED \
#   --db-path $DB_PATH \
#   --sim-threshold $SIM --top-k $TOPK \
#   --use-cot-compressed \
#   --max-trace-len 4000 \
#   --output $OUTDIR/ablation_rag_cot_compressed_65k.jsonl

echo ""
echo "============================================================"
echo " All 5 ablations complete. Results:"
echo "============================================================"
for f in $OUTDIR/ablation_*_65k.jsonl $OUTDIR/ablation_*_24k.jsonl; do
  n=$(wc -l < "$f")
  correct=$(python -c "
import json
recs=[json.loads(l) for l in open('$f') if l.strip()]
print(sum(1 for r in recs if r['is_correct']))
")
  echo "  $(basename $f): $correct/$n"
done
