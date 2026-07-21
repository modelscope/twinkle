#!/bin/bash
# train_skill_v2.sh — 简化 GRPO + buffer distill 训练启动脚本
# 用法: bash cookbook/exp/embedding/train_skill_v2.sh
#
# 环境变量:
#   LLM_BACKUP_API_KEY  - rubric 诊断用的教师 API key（必须，否则 buffer B 蒸馏不可用）
#   LLM_BACKUP_BASE_URL - 教师 API base URL
#   LLM_BACKUP_MODEL    - 教师模型 ID
#   GEN_MODEL_ID        - 训练 skill 模型 ID（默认 Qwen/Qwen3-4B）
#   TRAIN_GPUS / REF_GPUS / SKILL_SAMPLER_GPUS / BASE_SAMPLER_GPUS — GPU 分配

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./output/skill_v2}"

# 去重/排斥数据（冷启动 SFT 数据避免重叠）
EXCLUDE="${EXCLUDE_DATA_IDS:-}"

# 提前建目录：tee 需在 python 建目录前就能打开日志文件
mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/train_skill_v2.py" \
    --dataset aops \
    --n 20000 \
    --numeric-only \
    --eval-size 200 \
    --eval-every 10 \
    --chunk-size 32 \
    --n-skills 8 \
    --skill-retries 2 \
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
    --sft-weight 1.0 \
    --rubric-workers 16 \
    --sft-batch-size 8 \
    --ppo-mini-batch-size 0 \
    --grpo-epsilon 0.2 \
    --adv-clip 3.0 \
    --kl-beta 0.001 \
    --lr 1e-6 \
    --max-train-rounds 1500 \
    --save-rounds 200 \
    --output-dir "${OUTPUT_DIR}" \
    --swanlab-project twinkle \
    --swanlab-exp "skill_v2_$(date +%Y%m%d_%H%M%S)" \
    ${EXCLUDE:+--exclude-data-ids "${EXCLUDE}"} \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/run.log"
