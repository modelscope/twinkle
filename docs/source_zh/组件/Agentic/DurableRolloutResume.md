# Agentic RL 训练恢复

异步 Agentic RL 中，训练器取走一批完整 group 时，rollout 侧通常还有若干 query 正在执行
多轮工具调用。如果进程在这些 query 完成前退出，只恢复模型参数和 dataloader 游标会丢失
已经消费的 prompt。

本方案让 `StreamingDriver` 进入模型 checkpoint 的 `component_states`，只保存原始 query
和调度游标。它不复用 Env 快照，也不保存半条 trajectory、工具结果或 turn journal。

## 设计边界

- `StreamingDriver` 只负责自己的状态。checkpoint engine、sampler 和 Env 不替它维护状态。
- 模型、adapter、optimizer、scheduler、RNG 与 driver 状态由 `model.save()` 写入同一个
  checkpoint。
- `sync_weights()` 保持原有接口，不要求训练脚本传入 `checkpoint_ref` 或理解提交协议。
- 恢复时先加载模型 checkpoint，再把保存的 query 作为新的 group 从头执行。
- 当前实现不依赖 `LocalEnv`、`MultiTurnRollout` 或 `ledger` 的恢复接口。
- 保存位置沿用模型现有的本地目录、OSS 或 S3 存储实现。

## 保存

每个优化器 step 完成后执行：

```python
component = driver.state_dict()
checkpoint = model.save(
    f"policy-step-{step}",
    output_dir=str(checkpoint_root),
    adapter_name=adapter_name,
    save_optimizer=True,
    component_states={"streaming_driver": component},
)
ckpt_manager.sync_weights()
```

`state_dict()` 包含以下信息：

| 字段 | 含义 |
| --- | --- |
| `queries` | 尚未被新 checkpoint 覆盖的原始 query，以及仍在运行或等待消费的 query |
| `admitted_through` | 本次快照覆盖的最后一个 admission 序号 |
| `handed` | 已交给训练器的 batch 数 |
| `prompts_drawn` | 已从 dataloader 读取的 prompt 数 |
| `policy_version` | 快照时 driver 观察到的策略版本 |
| `loader_state` | dataloader 自带的状态；不存在时使用 `prompts_drawn` 恢复游标 |

`state_dict()` 是一次性 checkpoint 快照：生成后即释放已交付 group 的 query 历史。若
`model.save()` 抛出异常，训练进程应按上一个 checkpoint 恢复；同一进程若要重试，应直接
复用刚返回的 `component` 对象。仍在运行或尚未消费的 group 会继续出现在后续快照中。

## 恢复

先恢复模型，再恢复 driver：

```python
progress = model.resume_from_checkpoint(checkpoint, adapter_name=adapter_name)
component_states = progress.get("component_states") or {}

driver.load_state_dict(component_states.get("streaming_driver"))
ckpt_manager.sync_weights()
```

`load_state_dict()` 会重新提交保存的原始 query。已经生成过一部分轨迹的 group 也是从
第一条用户消息重新开始，因此恢复逻辑与具体 Env 无关。

## 一致性说明

模型 checkpoint 成功写入时，query 至少可以被恢复一次。若 checkpoint 恰好发生在一次
非幂等工具调用之后、group 完成之前，从原始 query 重放可能再次执行该工具。生产环境中的
外部工具应使用业务幂等键，或避免在不可重放操作尚未完成时触发 checkpoint。

本方案不保证半轨迹级 exactly-once，也不承诺保留已经生成的 token。它保证的是：

1. 权重更新不会把正在执行工具的 group query 直接删除；
2. 模型 checkpoint 与 driver 控制状态使用同一个原子保存入口；
3. 恢复后不会跳过已经消费但尚未训练的 prompt；
4. 默认 `resume` 关闭时不改变原有采样、准入和 staleness 语义。

## 示例

完整 GSM8K 双卡/多卡示例见：

```bash
MODEL_GPUS=4 SAMPLER_GPUS=4 MAX_STEPS=100 \
  bash cookbook/rl/async_rl/durable_multi_turn_grpo.sh
```

再次运行同一命令会加载最新 checkpoint，并恢复 checkpoint 中尚未完成的 query。
