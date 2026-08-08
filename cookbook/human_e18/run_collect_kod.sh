#!/bin/bash
# KodCode 冷启动数据采集：8 卡全 rollout，无 SFT，narrative prompt 不变。
cd /mnt/data/yzhao/tastelikefeet/twinkle/cookbook/human_e18
export PYTHONPATH=/mnt/data/yzhao/tastelikefeet/twinkle/src:${PYTHONPATH}
export SKILL_SAMPLER_GPUS=2 BASE_SAMPLER_GPUS=6   # token 预算比 16.5:1，2+6 实测理论最优
export KOD_SELFCHECK=0                            # 不自检（坏题采集时自然不入池）
export TARGET_SAMPLES=5000
exec /usr/local/bin/python -u e18_collect_kod.py
