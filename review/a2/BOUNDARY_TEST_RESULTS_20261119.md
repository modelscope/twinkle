# RolloutResume 边界测试结果

日期：2026-11-19

## 结论

八组边界均已有自动化测试或独立探针：

| 组别 | 结果 | 证据 |
|---|---|---|
| 1. 无故障等价与开销 | 通过 | 确定性 A/B 训练信号和评测样本逐值一致；journal/object 吞吐与空间放大已测量 |
| 2. trainer checkpoint 事务 | 通过 | 真实 `torch.optim.SGD` step、checkpoint 原子写、commit、回滚和 lease 清理 |
| 3. 完整进程强杀阶段 | 通过 | A2 Linux 五个阶段均以 `SIGKILL(-9)` 终止并正确恢复 |
| 4. exactly-once 边界 | 通过 | 缺失 checkpoint ref、重复 commit、已提交后旧快照、旧 policy 组均有测试 |
| 5. journal 故障注入 | 通过 | 截断、中间损坏、SHA 不匹配、ENOSPC、rename 中断、线程/进程并发 writer |
| 6. 环境 fail-closed | 通过 | 无恢复能力 Env、AgentEnv、OpenEnv、OpenEnvClient 均不允许静默恢复 |
| 7. 非幂等工具 | 通过 | 工具 intent 未提交时恢复拒绝，已完成 effect 使用缓存 observation，不重复调用 |
| 8. adapter 生命周期 | 通过 | 缺失路径拒绝恢复，同 version 不同路径固定原 adapter，过期组整组淘汰 |

本地聚焦测试：`33 passed`。

A2 Python 3.10 环境聚焦测试：`33 passed`。

## 无故障对照

确定性控制配置：

- `fault_period=0`
- 8 个训练组
- LoRA dropout `0`
- `torch.use_deterministic_algorithms(True, warn_only=True)`
- 16 条固定测试

结果：

- baseline/durable 均为 8 次 optimizer update、0 丢失组。
- 两组训练信号逐值一致。
- 16 条评测样本逐值一致。
- `accuracy_delta=0`，`reward_delta=0`。

随机采样小样本无故障对照仍观察到非确定性指标波动，但训练步数和丢失组完全一致。
因此精确等价结论只使用确定性控制；随机模式不能用于逐值比较。

## 性能与空间开销

A2、Python 3.10、500 次 journal append：

| 指标 | 结果 |
|---|---:|
| durable journal | `5.1788 s` |
| 单次 append | `10.358 ms` |
| journal 吞吐 | `96.55 events/s` |
| journal bytes/event | `530.56 B` |
| 重开并扫描 500 条 | `18.60 ms` |

50 个对象保存：

| 指标 | 结果 |
|---|---:|
| durable object save | `0.02754 s` |
| 单对象 | `0.551 ms` |
| 对象吞吐 | `1815.7 objects/s` |
| 存储放大 | `1.0326x` |

4 个独立进程各写 50 条：

- 共 200 条记录。
- sequence 连续为 `0..199`。
- hash chain 有效。
- 总耗时 `9.352 s`。

结论：单事件 fsync 成本明显，约 10 ms；批量高频小 journal 记录会成为吞吐瓶颈。
后续若接入真实训练，需要按 turn/score/lease 等关键边界写入，而不是每个细粒度事件
都同步 fsync。

## SIGKILL 阶段矩阵

A2 Linux 实测，子进程返回码均为 `-9`：

| 阶段 | 恢复结果 |
|---|---|
| prompt 已分配 | prompt/group/state_key 恢复 |
| group 已 ready | 重新交付 2 行、policy version 0 |
| batch 已 lease | 相同 lease 重新交付 |
| optimizer 已 step、checkpoint 未提交 | 回退上一 checkpoint，同一 batch 再交付 |
| checkpoint 已提交 | lease 清空，batch 不再交付 |

## 真实 Qwen + NPU checkpoint 强杀

在上述轻量事务探针之外，补充了真实 Qwen3-0.6B LoRA 训练进程验证：

- A2 Ascend 910B4、32 GB HBM、单卡 `npu:0`；
- 官方 GSM8K 训练记录，真实 Qwen forward/backward 和 AdamW step；
- 分别强杀 optimizer step 后、临时 checkpoint 保存后、checkpoint 原子发布后和
  journal commit 后；
- 4 个场景的 worker 返回码均为 `-9`，恢复验证全部通过；
- 3 个未提交边界恢复 `step-0` 并重新交付同一 lease；
- commit 后恢复 `step-1`、optimizer step `[1]`，lease 清空且不再交付。

详细结果见：

```text
review/a2/NPU_CHECKPOINT_KILL_RESULTS_20261119.md
review/a2/npu-checkpoint-kill-results/npu-checkpoint-kill-20261119/report.json
```

## 故障注入

- journal 最后一行截断：只删除 torn tail。
- 中间记录损坏：`JournalCorrupt` fail closed。
- 对象 SHA 不匹配：拒绝加载。
- `fsync` 返回 `ENOSPC`：append 回滚到原文件长度。
- `os.replace` 中断：旧对象保持完整，临时文件清理。
- 多线程/多进程 journal writer：共享文件锁并重新扫描，sequence 与 hash chain 连续。

本轮修复了两个实际缺陷：

1. `DurableJournal` 原锁仅限单实例，多 writer 会生成分叉链；现增加线程锁与跨进程
   文件锁，并在 append 前重新扫描。
2. append 在 fsync/磁盘错误后没有回滚；现恢复原文件长度，避免“调用报错但记录已写入”。

## 产物位置

本地 Windows 探针结果：

```text
review/a2/boundary-results/checkpoint-transaction.json
review/a2/boundary-results/kill-phases-windows.json
review/a2/boundary-results/overhead-windows.json
review/a2/boundary-results/focused-tests-local.log
```

A2 同步结果：

```text
review/a2/boundary-results/a2-20261119/
```

关键文件：

```text
boundary/focused-tests.log
boundary-source-hashes.txt
boundary-kill-phases.json
boundary-checkpoint-transaction.json
boundary-overhead.json
boundary-no-fault-deterministic/report.json
```

## 未覆盖的真实线上边界

- AgentEnv、OpenEnv、OpenEnvClient 本组只验证了无恢复能力时 fail closed；真实沙箱
  suspend/attach 服务协议未测试。
- 磁盘满使用 `ENOSPC` 注入，没有实际写满测试盘。
- checkpoint 事务已使用真实 Qwen3-0.6B LoRA/NPU，但尚未接入完整 Twinkle training
  cookbook 的 optimizer、scheduler、DataLoader 全量状态。
- 性能数据是单机本地文件系统结果，不代表网络存储、分布式文件系统或高并发多卡。
