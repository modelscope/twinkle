# GRPOAdvantage

GRPO (Group Relative Policy Optimization) 优势函数通过减去组内均值来计算优势。

## 使用示例

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# 假设有 2 个 prompt,每个生成 4 个样本
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 个奖励值
advantages = advantage_fn(rewards, num_generations=4, scale='none')

# advantages 会是每组减去组内均值:
# 第一组: [0.0-0.5, 1.0-0.5, 0.0-0.5, 1.0-0.5] = [-0.5, 0.5, -0.5, 0.5]
# 第二组: [1.0-0.25, 0.0-0.25, 0.0-0.25, 0.0-0.25] = [0.75, -0.25, -0.25, -0.25]
```

## 工作原理

GRPO 将样本分组(每组对应一个 prompt 的多个生成),然后在组内:
1. 计算组内奖励均值
2. 每个样本的优势 = 该样本的奖励 - 组内均值
3. 可选地对优势值进行归一化

这种方法能够:
- 减少方差,提高训练稳定性
- 在组内进行相对比较,更符合人类偏好的相对性
- 避免奖励尺度的影响

## 训练集成

通过 `SamplingParams(num_samples=N, logprobs=1)` 设置采样数量，从每个响应的 `sequences` 读取生成结果。训练还需要对齐的 labels 和旧策略 logprobs，仅传 advantages 不足以完成正确的训练。

完整的采样、奖励计算和优化器循环见 [GRPO 训练示例](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/short_math_grpo.py)。使用 RLOO 时，将优势函数替换为 `RLOOAdvantage`，并为每个 prompt 至少生成两个样本。
