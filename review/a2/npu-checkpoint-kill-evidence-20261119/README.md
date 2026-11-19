# Qwen NPU Checkpoint 强杀证据摘要

本目录只保留可审查的 JSON 和日志，不包含 LoRA adapter、optimizer checkpoint、
`.resume` 对象或 tar 包。

四个场景均为真实 Qwen3-0.6B LoRA/NPU 训练进程 `SIGKILL`：

| 场景 | worker 返回码 | 恢复 checkpoint | lease 重交付 |
|---|---:|---|---|
| `optimizer_step_uncommitted` | `-9` | `step-0` | 2 行 |
| `adapter_temp_uncommitted` | `-9` | `step-0` | 2 行 |
| `checkpoint_promoted_uncommitted` | `-9` | `step-0` | 2 行 |
| `checkpoint_committed` | `-9` | `step-1` | 0 行 |

汇总报告：`report.json`

完整本地证据目录（不提交 Git）：

```text
review/a2/npu-checkpoint-kill-results/npu-checkpoint-kill-20261119/
```

完整结果包 SHA-256：

```text
4CCBA9F9B55657E9288838E8C58E489DC19815F07C94E863869D3481DC6EF968
```

复现脚本：

```text
review/a2/run_npu_checkpoint_kill_probe.py
```
