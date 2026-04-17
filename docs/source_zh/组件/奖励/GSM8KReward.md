# GSM8K 奖励

专为评估 GSM8K 数学问题求解设计的奖励函数。

## GSM8KAccuracyReward

通过提取 boxed 或 hash 格式（`####`）的答案并进行数值/字符串比较来评估 GSM8K 答案的正确性。

```python
from twinkle.reward import GSM8KAccuracyReward

reward_fn = GSM8KAccuracyReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 表示正确, 0.0 表示错误
```

奖励函数的工作流程:
1. 从模型补全中提取 `\boxed{...}` 或 `#### ...` 格式的答案
2. 从参考轨迹中提取真实答案
3. 执行数值比较（带容差）或精确字符串匹配

## GSM8KFormatReward

检查模型输出是否包含正确格式的答案部分。

```python
from twinkle.reward import GSM8KFormatReward

reward_fn = GSM8KFormatReward()
rewards = reward_fn(trajectories, ground_truths)
# rewards: List[float], 1.0 表示格式有效, 0.0 表示无效
```

> 在数学问题求解的 GRPO 训练中，将 GSM8KAccuracyReward 和 GSM8KFormatReward 组合使用作为复合奖励。
