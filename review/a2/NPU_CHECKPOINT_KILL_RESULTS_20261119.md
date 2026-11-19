# Qwen NPU Checkpoint 强杀验证结果

日期：2026-11-19

## 结论

真实 Qwen3-0.6B LoRA 训练进程在 4 个 checkpoint 事务边界执行 Linux `SIGKILL`，
全部恢复成功：

| 强杀边界 | 进程返回码 | 恢复 checkpoint | optimizer step | lease 重交付 |
|---|---:|---|---:|---|
| optimizer step 后、adapter 保存前 | `-9` | `step-0` | 空 | 同一 lease，2 行 |
| 临时 adapter/optimizer 保存后、发布前 | `-9` | `step-0` | 空 | 同一 lease，2 行 |
| checkpoint 原子发布后、journal commit 前 | `-9` | `step-0` | 空 | 同一 lease，2 行 |
| checkpoint journal commit 后 | `-9` | `step-1` | `[1]` | 不重交付 |

该结果证明 checkpoint、optimizer state、batch lease 和 commit journal 在真实 NPU
训练进程被强杀后保持一致的提交/回退语义。

## 环境

- 主机：A2 `e1f3eeac3fc6`
- 设备：Ascend 910B4，32 GB HBM，单卡 `npu:0`
- Python：3.10.12
- torch：2.10.0+cpu
- torch_npu：2.10.0
- transformers：5.20.2
- peft：0.21.3
- 模型：`/workspace/shared_assets/models/Qwen/Qwen3-0.6B`
- 训练：LoRA rank 8，BF16，真实 Qwen forward/backward，AdamW optimizer step
- 数据：官方 GSM8K `train.jsonl` 第一条记录

为保证 GRPO 组存在非零 advantage，探针使用 GSM8K 标准问题及固定正确/错误 completion
对。模型前向、反向、AdamW 更新、LoRA adapter 保存、optimizer checkpoint 和进程
SIGKILL 都是真实执行，不是 mock。

## 事务判定

### 未提交边界

以下三个边界恢复后都选择上一份 `step-0` checkpoint：

1. 内存中已经完成 optimizer step，但尚未保存。
2. 新 adapter 和 optimizer 已完整写入临时目录，但尚未原子发布。
3. 新 checkpoint 已原子发布为 `step-1`，但 journal 尚未记录 commit。

三者共同验证：

- 恢复后的 LoRA 权重 digest 等于 `step-0`。
- optimizer state 为空，未泄漏内存中那一次 AdamW step。
- 同一个 batch lease 被重新交付 2 行。
- 临时或已发布但未提交的 `step-1` 不会成为恢复依据。

### 已提交边界

journal commit 后 `SIGKILL`，恢复结果：

- 选择 journal 中登记的 `step-1`。
- LoRA digest 与训练后内存权重完全一致。
- optimizer step 恢复为 `[1]`。
- lease 已清除。
- 同一 batch 不再交付，`redelivered_rows=0`。

## 产物

探针：

```text
review/a2/run_npu_checkpoint_kill_probe.py
```

完整结果：

```text
review/a2/npu-checkpoint-kill-results/npu-checkpoint-kill-20261119/
```

PR 可审查摘要：

```text
review/a2/npu-checkpoint-kill-evidence-20261119/
```

关键文件：

```text
report.json
run.log
*/ready.json
*/worker.log
*/verify.json
*/verify.log
*/checkpoints/step-0/
*/checkpoints/step-1/
*/resume/rollout.jsonl
*/resume/objects/
```

打包文件：

```text
review/a2/npu-checkpoint-kill-results/npu-checkpoint-kill-20261119.tar.gz
```

SHA-256：

```text
探针：A082C5A67037FE305B98CA91FE4DCCA493A762C24C44103E3B3E65394A615EAB
结果包：4CCBA9F9B55657E9288838E8C58E489DC19815F07C94E863869D3481DC6EF968
```

本地与 A2 上的探针 SHA-256 一致。

## 尚未覆盖

- checkpoint 目录位于 A2 本地 overlay 文件系统，不是可写 NFS、GlusterFS 或对象存储。
- 当前是单卡 LoRA，不是多卡 HCCL/FSDP2 或全参数训练。
- 当前探针使用真实训练 step 和真实 checkpoint，但尚未接入完整 Twinkle training
  cookbook 的 optimizer/scheduler/DataLoader 全量状态。
- 未验证网络存储延迟、超时、连接重置和对象存储最终一致性。
