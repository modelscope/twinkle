# GRPOAdvantage

GRPO (Group Relative Policy Optimization) 优势函数通过减去组内均值来计算优势。

## 使用示例

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# 假设有 2 个 prompt,每个生成 4 个样本
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 个奖励值
advantages = advantage_fn(rewards, num_generations=4, scale='group')

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

## 完整训练示例

在 GRPO 训练中使用优势函数:

```python
from twinkle.advantage import GRPOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler

# Create components
actor = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B')
sampler = vLLMSampler(model_id='ms://Qwen/Qwen3.5-4B')
reward_fn = ...
advantage_fn = GRPOAdvantage()

# Training loop
for batch in dataloader:
    # Sample generation
    sample_response = sampler.sample(batch, num_samples=4)
    input_data = [seq.new_input_feature for response in sample_response for seq in response.sequences]
    ...
    rewards = reward_fn(...)

    # Calculate advantages
    advantages = advantage_fn(rewards, num_generations=4)

    # 4. Policy optimization
    loss = actor.forward_backward(
        inputs=input_data,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> GRPO 方法简单高效,适合大多数 RLHF 训练场景。
