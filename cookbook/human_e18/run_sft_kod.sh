#!/bin/bash
# E18-KOD 离线 SFT + 首尾 eval：训一遍，看 loss 收敛 + 训练前后 lift 差异。
# ⚠️ 会占满 8 卡（4 训练 + 2 skillmodel + 2 executor）—— 采集进程若还在跑，先确认它已停。
# ⚠️ eval 要跑 judge（本地 pytest 沙箱），不需要教师 API key。
cd /mnt/data/yzhao/tastelikefeet/twinkle/cookbook/human_e18
export PYTHONPATH=/mnt/data/yzhao/tastelikefeet/twinkle/src:${PYTHONPATH}
export TRAIN_GPUS=4            # 4 训练
export SKILL_SAMPLER_GPUS=2    # 2 张出 skill
export BASE_SAMPLER_GPUS=2     # 2 张跑 executor
export BATCH_SIZE=16           # 必须是 TRAIN_DP(=4) 的整倍数
export MICRO_BATCH=8
export LR=1e-5                 # 与在线版一致：恒定 lr、无 warmup/decay
export EPOCHS=3
export EVAL_SIZE=100           # 首尾各 100 题，选自 KodCode 中未生成过 skill 的题
export EXEC_ROLLOUTS=1         # executor 单次
export EXEC_TEMPERATURE=0.0    # greedy，可复现
export SAVE_EVERY_STEPS=0      # 0 = 只在结束时存权重
exec /usr/local/bin/python -u e18_sft_kod.py
