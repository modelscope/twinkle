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
from twinkle.reward import MathReward

# 创建组件
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

# 训练循环
for batch in dataloader:
    # 1. 采样生成
    response = sampler.sample(batch, num_samples=4)
    
    # 2. 计算奖励
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. 计算优势
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. 策略优化
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> GRPO 方法简单高效,适合大多数 RLHF 训练场景。
