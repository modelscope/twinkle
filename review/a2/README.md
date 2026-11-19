# RolloutResume A2 验证

验证目标分为两层：先用组收益探针确认真实故障窗口下可训练组被保留，再用 NPU 探针确认采样真值可恢复：

- 已采样 token、labels、completion mask 和 old logprob 保持完全一致；
- LocalEnv workspace 在第二进程重新分配 session 后仍可读取；
- Qwen3-0.6B 在 Ascend 910B4 上继续下一轮真实采样；
- 普通游标重启丢失的半完成组由 durable resume 恢复为同策略完整组；
- checkpoint commit 后同一 batch 不再重复交付；
- 不修改 `/workspace/codex-twinkle-20261115/twinkle` 等已有源码目录。

执行前先加载 CANN，并关闭 torch 后端自动加载：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export STAGE=/workspace/<isolated-rollout-resume-dir>
export PYTHONPATH=$STAGE/src:$STAGE/review/gate0
cd $STAGE
/workspace/npu-ai-env/bin/python review/a2/run_focused_tests.py
/workspace/npu-ai-env/bin/python review/a2/run_group_yield_probe.py
/workspace/npu-ai-env/bin/python review/a2/run_npu_resume_probe.py \
  --root $STAGE/results/npu-probe
/workspace/npu-ai-env/bin/python review/a2/run_npu_checkpoint_kill_probe.py \
  --root $STAGE/results/npu-checkpoint-kill-20261119 \
  --train-jsonl $STAGE/data/gsm8k/train.jsonl
```

成功标准：父进程观察到第一阶段返回 `-9/137`，第二阶段输出 JSON 中
`restored_exact=true`、`tool_observation=turn-1-durable`，且两轮 logprob 都是有限值。
checkpoint 强杀探针的 4 个场景都应满足 `worker_returncode=-9` 和 `report.json`
中的 `ok=true`。
