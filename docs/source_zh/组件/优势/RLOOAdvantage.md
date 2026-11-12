# RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) 优势函数使用留一法计算基线。

## 使用示例

```python
from twinkle.advantage import RLOOAdvantage

advantage_fn = RLOOAdvantage()

rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
advantages = advantage_fn(rewards, num_generations=4, scale='none')

# 对于每个样本,基线是除了它以外的其他样本的均值
# 第一组第一个样本: 0.0 - mean([1.0, 0.0, 1.0]) = 0.0 - 0.667 = -0.667
# ...
```

## 工作原理

RLOO 对每个样本:
1. 计算除该样本外组内其他样本的奖励均值 (留一基线)
2. 优势 = 该样本奖励 - 留一基线
3. 可选地进行归一化

RLOO 的优势:
- 避免使用样本自身信息作为基线,减少偏差
- 更准确地估计反事实基线
- 在样本数量较多时效果更好

## 训练集成

通过 `SamplingParams(num_samples=N, logprobs=1)` 设置采样数量，从每个响应的 `sequences` 读取生成结果。训练还需要对齐的 labels 和旧策略 logprobs，仅传 advantages 不足以完成正确的训练。

完整的采样、奖励计算和优化器循环见 [GRPO 训练示例](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/short_math_grpo.py)。使用 RLOO 时，将优势函数替换为 `RLOOAdvantage`，并为每个 prompt 至少生成两个样本。
