#!/bin/bash
# train_skill_v2_ablate3.sh — 三路 prompt×thinking 消融，各 40 GRPO rounds，顺序执行。
#
# 三组（单变量：skill 文体 + thinking）：
#   1) narrative + think   -> output.ablate_narrative_think/skill_v2
#   2) pitfall   + think   -> output.ablate_pitfall_think/skill_v2
#   3) pitfall   + nothink -> output.ablate_pitfall_nothink/skill_v2
#
# 关键设计（对应探针结论 + 训练脚本代码事实）：
#   A. 离线 SFT/buffer B 暂时关闭：本脚本 unset LLM_BACKUP_*/OPENAI 环境，使
#      build_rubric_checker() 返回 None -> 训练走 "GRPO only"（train_skill_v2.py:1596-1597,1673）。
#      理由：40 rounds×chunk16≈640 题，远够不到 distill-trigger(150) 与 sft-trigger(100)，
#      SFT 本就不会触发；关掉还能省去每条失败轨迹的 qwen-plus 预诊断 API 开销与后台线程。
#   B. think 组把 --skill-max-tokens 覆盖回 8192：基础 .sh 写死 4096，装不下 think 段+完整
#      <skills>，会截断成空块导致 parse 崩（train_skill_v2.py:1455-1457）。nothink 组维持 4096。
#   C. eval 保留（--eval-every/--eval-size 沿用基础 .sh 的 5/200），这是三组对比的产出信号，不能关。
#
# ⚠ 重要提醒（务必在 swanlab 盯 leak/rate 曲线）：
#   主 GRPO 路径 reward = parseable AND correct，leak 仅作 observability 审计、不进 reward
#   （train_skill_v2.py:1195-1197,1208）。buffer B 关闭后训练回路里【没有任何泄漏防御】。
#   探针实测 narrative+think 净泄漏≈0.46、pitfall+think 也偏高 —— 两个 think 组极可能
#   reward-hacking：靠把答案写进 skill 拿高 reward，reward_mean/eval-lift 会虚高。
#   解读 think 组时必须同时看 leak/rate；pitfall+nothink 泄漏≈0，是干净对照基线。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A. 关闭 buffer B / 离线 SFT / rubric 预诊断（GRPO only）
unset LLM_BACKUP_API_KEY LLM_BACKUP_BASE_URL LLM_BACKUP_MODEL OPENAI_API_KEY || true

ROUNDS="${ROUNDS:-40}"

# 每行: TAG STYLE THINKING(on/off) SKILL_MAX_TOKENS
RUNS=(
    "narrative_think   narrative on  8192"
    "pitfall_think     pitfall   on  8192"
    "pitfall_nothink   pitfall   off 4096"
)

for spec in "${RUNS[@]}"; do
    read -r TAG STYLE THINKING SMT <<< "${spec}"
    OUT="./output.ablate_${TAG}/skill_v2"
    echo "=============================================================="
    echo "[ablate3] TAG=${TAG} style=${STYLE} thinking=${THINKING} skill_max_tokens=${SMT} rounds=${ROUNDS}"
    echo "  输出目录: ${OUT}   (GRPO only, buffer B 已关)"
    echo "=============================================================="
    OUTPUT_DIR="${OUT}" \
    bash "${SCRIPT_DIR}/train_skill_v2.sh" \
        --skill-style "${STYLE}" \
        --skill-thinking "${THINKING}" \
        --skill-max-tokens "${SMT}" \
        --max-train-rounds "${ROUNDS}" \
        --swanlab-exp "ablate3_${TAG}_$(date +%Y%m%d_%H%M%S)"
    echo "[ablate3] ${TAG} 完成 -> ${OUT}"
    # 两次连跑之间等 Ray/vLLM 完全退出，避免引擎初始化竞态（曾复现过）
    sleep 30
done

echo "[ablate3] 三组全部完成："
echo "  narrative+think   -> ./output.ablate_narrative_think/skill_v2"
echo "  pitfall+think     -> ./output.ablate_pitfall_think/skill_v2"
echo "  pitfall+nothink   -> ./output.ablate_pitfall_nothink/skill_v2"
