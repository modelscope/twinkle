# CompletionRewardMetric

The CompletionRewardMetric aggregates key statistics during RLHF training, including generation time, weight synchronization time, reward scores, and completion lengths.

```python
from twinkle.metric import CompletionRewardMetric

metric = CompletionRewardMetric(device_mesh=..., process_group=...)

# Accumulate during training loop
metric.accumulate(
    inputs,
    outputs,
    generation_time=gen_time,
    weight_sync_time=sync_time,
    rewards=reward_values,
    completions=completion_texts,
)

# Calculate aggregated metrics
result = metric.calculate()
# result contains: generation_time, weight_sync_time, mean_reward, mean_completion_length, etc.
```

This metric is designed for GRPO and other RL training loops where monitoring generation quality and system performance is essential.

> CompletionRewardMetric performs DP-aware aggregation, correctly averaging metrics across all data-parallel ranks.
