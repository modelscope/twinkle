# CompletionRewardMetric

CompletionRewardMetric 在 RLHF 训练过程中聚合关键统计数据，包括生成时间、权重同步时间、奖励分数和补全长度。

```python
from twinkle.metric import CompletionRewardMetric

metric = CompletionRewardMetric(device_mesh=..., process_group=...)

# 在训练循环中累积
metric.accumulate(
    inputs,
    outputs,
    generation_time=gen_time,
    weight_sync_time=sync_time,
    rewards=reward_values,
    completions=completion_texts,
)

# 计算聚合指标
result = metric.calculate()
# result 包含: generation_time, weight_sync_time, mean_reward, mean_completion_length 等
```

此指标专为 GRPO 和其他 RL 训练循环设计，用于监控生成质量和系统性能。

> CompletionRewardMetric 执行 DP 感知的聚合，在所有数据并行 rank 上正确地取平均值。
