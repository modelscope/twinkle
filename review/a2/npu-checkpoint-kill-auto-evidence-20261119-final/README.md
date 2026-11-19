# 自动持久化 NPU 强杀证据

本目录只保存不含模型权重的验证证据。完整运行日志位于 A2 验证目录，被仓库的
`*.log` 规则排除。

四个场景均由真实 Qwen3-0.6B、LoRA、Ascend 910B4 和 `SIGKILL` 产生：

| 场景 | 返回码 | lease 重交付 | optimizer step |
|---|---:|---:|---:|
| optimizer step 后、保存前 | `-9` | 是 | `[]` |
| 临时 checkpoint 保存后 | `-9` | 是 | `[]` |
| checkpoint 发布后、commit 前 | `-9` | 是 | `[]` |
| commit 后 | `-9` | 否 | `[1]` |

`report.json` 聚合四个场景；每个场景目录下的 `ready.json` 和 `verify.json`
分别记录强杀前事务状态与重启后的 LoRA/optimizer 校验。
