# Agentic RL 训练恢复

异步 Agentic RL 中，训练器取走一批完整 group 时，rollout 侧通常还有若干 query 正在执行
多轮工具调用。如果进程在这些 query 完成前退出，只恢复模型参数和 dataloader 游标会丢失
已经消费的 prompt。

本方案让 `StreamingDriver` 进入模型 checkpoint 的 `component_states`，只保存原始 query
和调度游标。它不复用 Env 快照，也不保存半条 trajectory、工具结果或 turn journal。

## 设计边界

- `StreamingDriver` 与 checkpoint manager 各自维护自己的组件状态，sampler 和 Env 不替它们维护状态。
- 模型、adapter、optimizer、scheduler、RNG 与上述组件状态由 `model.save()` 统一写入同一个
  checkpoint。
- `sync_weights()` 保持原有接口，不要求训练脚本传入 `checkpoint_ref` 或理解提交协议。
- 恢复时先加载模型 checkpoint，再恢复 checkpoint manager 并同步权重，最后把保存的 query
  作为新的 group 从头执行。
- 当前实现不依赖 `LocalEnv`、`MultiTurnRollout` 或 `ledger` 的恢复接口。
- 保存位置沿用模型现有的本地目录、OSS 或 S3 存储实现。

## 保存

每个优化器 step 完成后执行：

```python
component_states = {
    "streaming_driver": driver.state_dict(),
    "checkpoint_manager": ckpt_manager.state_dict(),
}
checkpoint = model.save(
    f"policy-step-{step}",
    output_dir=str(checkpoint_root),
    adapter_name=adapter_name,
    save_optimizer=True,
    component_states=component_states,
)
ckpt_manager.sync_weights(merge_and_sync=False)
```

`merge_and_sync=False` 的 LoRA 同步分为两个阶段：第一次同步基模，第二次同步 adapter。
训练开始和 checkpoint 恢复后都必须先调用两次完成预热，再让 driver 提交 query；之后每个
optimizer step 只需同步一次 adapter。示例脚本通过 `warm_policy_weights()` 封装这两步。

`StreamingDriver.state_dict()` 包含以下信息：

| 字段 | 含义 |
| --- | --- |
| `queries` | checkpoint 时仍在运行或等待消费的原始 query |
| `pending_rows` | DataLoader 已 yield 整批、但 driver 尚未 admit 的原始行 |
| `handed` | 已交给训练器的 batch 数 |
| `prompts_drawn` | 已 admit 的 prompt 数，不包含 `pending_rows` |
| `policy_version` | 快照时 driver 观察到的策略版本 |
| `base_version` | admission window 的起始策略版本 |
| `consumed_batches` | 训练器已取走的 batch 确认数 |
| `live` | 累计已准入且未被淘汰的 member 数，包含成功交付的历史 |
| `active_live` | 当前持久化 query 已占用的 member 数 |
| `loader_state` | dataloader 自带的状态；不存在时使用 `prompts_drawn + len(pending_rows)` 恢复 plain iterable 游标 |

上述状态快照不修改 admission、staleness 或采样调度状态。
已经交给训练器的 batch 不写入下一个 checkpoint；若新 checkpoint 写入失败，从上一个
checkpoint 恢复时仍会重放该 batch。

### 多模态状态

`component_states` 最终写入 `trainer_state.json`，因此状态编码保持为 JSON，不使用 pickle。
`queries` 和 `pending_rows` 中的以下值会透明往返：

- `torch.Tensor`：保存 dtype、shape 和 C-order 原始字节，加载后恢复 tensor；
- `numpy.ndarray`：保存 dtype、shape 和原始字节；
- `PIL.Image.Image`：保存为 PNG 字节；
- `bytes` / `bytearray`：base64 编码；
- 嵌套的 list、tuple、mapping 和 path。

较大的二进制 payload 只有在 zlib 压缩后更小时才启用压缩。`pixel_values`、`image_grid_thw`、
`video_grid_thw` 以及 HF 数据集常见的 `{"bytes": ..., "path": ...}` 图片行都可以直接保存。
该方案会增加 `trainer_state.json` 的大小，但保持 checkpoint 自包含，不需要额外 sidecar
协议或改变 `model.save()` 接口。

`live` 是累计 admission 计数，不是只统计当前仍在运行的 episode；成功完成并交付的历史
member 仍会影响后续准入窗口。恢复时先回到 `live - active_live` 的历史基线，再由
`load_state_dict()` 重新提交 query 并加回新的 active member，避免历史丢失或重复计数。
`base_version` 和 `consumed_batches` 同时恢复，保证重启前后的容量窗口口径一致。

## 恢复

先恢复模型，再恢复 checkpoint manager 并同步权重，最后恢复 driver：

```python
progress = model.resume_from_checkpoint(checkpoint, adapter_name=adapter_name)
component_states = progress.get("component_states") or {}
driver_state = component_states.get("streaming_driver")
manager_state = component_states.get("checkpoint_manager")

if manager_state is not None:
    ckpt_manager.load_state_dict(manager_state)
warm_policy_weights(ckpt_manager)
driver.load_state_dict(driver_state)
```

不包含 `streaming_driver` 的普通模型 checkpoint 仍可按原流程恢复，不要求 driver 组件状态。
`StreamingDriver.load_state_dict()` 会重新提交保存的原始 query。已经生成过一部分轨迹的 group
也是从第一条用户消息重新开始，因此恢复逻辑与具体 Env 无关。

## 一致性说明

模型 checkpoint 成功写入时，query 至少可以被恢复一次。若 checkpoint 恰好发生在一次
非幂等工具调用之后、group 完成之前，从原始 query 重放可能再次执行该工具。生产环境中的
外部工具应使用业务幂等键，或避免在不可重放操作尚未完成时触发 checkpoint。

本方案不保证半轨迹级 exactly-once，也不承诺保留已经生成的 token。它保证的是：

1. 权重更新不会把正在执行工具的 group query 直接删除；
2. 模型 checkpoint 与 driver、checkpoint manager 控制状态使用同一个原子保存入口；
3. 恢复后不会跳过已经消费但尚未训练的 prompt；
4. 不调用组件状态保存/加载时，不改变原有采样、准入和 staleness 语义。

## 示例

完整 GSM8K 双卡/多卡示例见：

```bash
MODEL_GPUS=4 SAMPLER_GPUS=4 MAX_STEPS=100 \
  bash cookbook/rl/async_rl/durable_multi_turn_grpo.sh
```

再次运行同一命令会加载最新 checkpoint，并恢复 checkpoint 中尚未完成的 query。
