#!/bin/bash
# train_skill_v2.sh — 简化 GRPO + buffer distill 训练启动脚本
# 用法: bash cookbook/exp/skill2lora/train_skill_v2.sh
#
# 环境变量:
#   LLM_BACKUP_API_KEY  - rubric 诊断用的教师 API key（必须，否则 buffer B 蒸馏不可用）
#   LLM_BACKUP_BASE_URL - 教师 API base URL
#   LLM_BACKUP_MODEL    - 教师模型 ID
#   GEN_MODEL_ID        - 训练 skill 模型 ID（默认 Qwen/Qwen3-4B）
#   TRAIN_GPUS / REF_GPUS / SKILL_SAMPLER_GPUS / BASE_SAMPLER_GPUS — GPU 分配

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 缓解显存碎片（reserved-but-unallocated），降低 forward_backward 阶段 OOM 概率
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 默认输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./output/skill_v2}"

# 去重/排斥数据（冷启动 SFT 数据避免重叠）
EXCLUDE="${EXCLUDE_DATA_IDS:-}"

# 与 SEAM 输入数据对齐：直读 SEAM build_aops_dataset.py 产出的 parquet（同题池 + 同 val）
# 置空则回退到 twinkle 自己的 load+shuffle+split。
SEAM_PARQUET_DIR="${SEAM_PARQUET_DIR:-/root/data/seam}"

# 提前建目录：tee 需在 python 建目录前就能打开日志文件
mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/train_skill_v2.py" \
    --dataset aops \
    --numeric-only \
    --eval-size 200 \
    --eval-every 5 \
    --eval-rollouts 1 \
    --eval-skill-temperature 0.0 \
    --chunk-size 16 \
    --n-skills 8 \
    --distill-retries 1 \
    --skill-gen-temperature 1.0 \
    --skill-gen-top-p 1.0 \
    --skill-gen-top-k -1 \
    --max-model-len 16384 \
    --max-tokens 8192 \
    --skill-max-tokens 4096 \
    --len-budget 600 \
    --distill-trigger 150 \
    --distill-batch 64 \
    --sft-trigger 100 \
    --passatk-k 8 \
    --passatk-m 2 \
    --align-mode seam \
    --sft-weight 1.0 \
    --rubric-workers 16 \
    --sft-batch-size 4 \
    --ppo-mini-batch-size 0 \
    --grpo-epsilon 0.2 \
    --adv-clip 0 \
    --kl-beta 0.001 \
    --lr 1e-6 \
    --max-train-rounds 1500 \
    --save-rounds 200 \
    --output-dir "${OUTPUT_DIR}" \
    --swanlab-project twinkle \
    --swanlab-exp "skill_v2_$(date +%Y%m%d_%H%M%S)" \
    ${EXCLUDE:+--exclude-data-ids "${EXCLUDE}"} \
    ${SEAM_PARQUET_DIR:+--seam-parquet-dir "${SEAM_PARQUET_DIR}"} \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"
