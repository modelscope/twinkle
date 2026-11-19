# 自动持久化版 NPU Checkpoint 强杀结果

## 目的

验证 `StreamingDriver` 在启用 `ResumeStore` 后能自行在稳定边界写入 driver
快照。测试进程不再调用 `driver.state_dict()`，避免人工快照掩盖自动持久化缺口。

## 环境

| 项目 | 配置 |
|---|---|
| 主机 | A2，`e1f3eeac3fc6` |
| NPU | Ascend 910B4，32768 MB HBM |
| Python | 3.10.12 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| transformers | 5.20.2 |
| peft | 0.21.3 |
| 模型 | Qwen3-0.6B |
| 训练方式 | LoRA rank 8，BF16，真实 optimizer step |
| 数据 | 官方 GSM8K `train.jsonl` 固定样本 |

## 四场景结果

四个 worker 均在目标边界被真实 `SIGKILL`，进程返回码均为 `-9`。

| 强杀边界 | lease 恢复 | 重交付行数 | optimizer step | LoRA digest | 结论 |
|---|---:|---:|---:|---:|---|
| optimizer step 后、adapter 保存前 | 是 | 2 | `[]` | 匹配 | 未提交更新不泄漏 |
| 临时 adapter/optimizer 保存后 | 是 | 2 | `[]` | 匹配 | 临时 checkpoint 不生效 |
| checkpoint 原子发布后、commit 前 | 是 | 2 | `[]` | 匹配 | 已发布 checkpoint 不被误提交 |
| commit 后 | 否 | 0 | `[1]` | 匹配 | 已提交 batch 不重复训练 |

所有四个场景均满足：

- `worker_returncode == -9`；
- `driver_persistence == automatic`；
- 恢复后的 LoRA digest 与预期 checkpoint 一致；
- 恢复后的 optimizer step 与事务边界一致；
- 未提交场景重交付同一 lease，已提交场景不再交付。

## 聚焦测试

本地和 A2 均通过：

```text
45 passed
```

测试覆盖自动 prompt 快照、ready group、batch lease、commit、旧快照升级、
policy binding 冲突拒绝、Env fail-closed、非幂等工具和 journal 故障注入。

## 证据

- `review/a2/npu-checkpoint-kill-auto-evidence-20261119-final/report.json`
- 每个场景的 `ready.json` 和 `verify.json`

完整 LoRA、optimizer 二进制和运行日志仍保留在 A2 验证目录，不进入 Git。
