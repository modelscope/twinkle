#!/bin/bash
# train_skill_v2_ablate.sh — skill 文体消融：toy vs pitfall，各 30 rounds，顺序执行
#
# 设计（对应探针实验结论 CONCLUSIONS_config.md / CONCLUSIONS_reflexion.md）：
#   1. thinking 可控（--skill-thinking），本轮两个实验均为 off；
#   2. 两个文体方向：toy（异数字玩具题示范）与 pitfall（预判纠错）；
#      同一文体在主链路（query-only）与 buffer B regen（rubric 诊断）下输出格式一致，
#      保证 GRPO 与 SFT 样本分布一致可联合训练；
#   3. 每个实验 --max-train-rounds 30 结束，输出分目录：
#      output.ablate_toy/skill_v2 与 output.ablate_pitfall/skill_v2
#
# 用法: bash cookbook/exp/embedding/train_skill_v2_ablate.sh
# 注：train_skill_v2.sh 末尾以 "$@" 透传附加参数，argparse 同名参数后者覆盖前者，
#     故此处的 --max-train-rounds 30 会覆盖基础脚本里的 1500。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for STYLE in toy pitfall; do
    echo "=============================================================="
    echo "[ablate] 开始 skill-style=${STYLE} (thinking=off, 30 rounds)"
    echo "=============================================================="
    OUTPUT_DIR="./output.ablate_${STYLE}/skill_v2" \
    bash "${SCRIPT_DIR}/train_skill_v2.sh" \
        --skill-style "${STYLE}" \
        --skill-thinking off \
        --max-train-rounds 30
    echo "[ablate] skill-style=${STYLE} 完成"
    # 中文注释：两次连跑之间等 Ray/vLLM 完全退出，避免引擎初始化竞态（曾复现过一次）
    sleep 30
done

echo "[ablate] 两个消融实验全部完成："
echo "  toy     -> ./output.ablate_toy/skill_v2"
echo "  pitfall -> ./output.ablate_pitfall/skill_v2"
