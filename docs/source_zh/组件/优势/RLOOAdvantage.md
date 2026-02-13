# RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) 优势函数使用留一法计算基线。

## 使用示例

```python
from twinkle.advantage import RLOOAdvantage

advantage_fn = RLOOAdvantage()

rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
advantages = advantage_fn(rewards, num_generations=4)

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

## 完整训练示例

```python
from twinkle.advantage import RLOOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward

# 创建组件
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = RLOOAdvantage()

# 训练循环
for batch in dataloader:
    # 1. 采样生成(每个 prompt 生成更多样本以提高 RLOO 效果)
    response = sampler.sample(batch, num_samples=8)
    
    # 2. 计算奖励
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. 计算优势
    advantages = advantage_fn(rewards, num_generations=8)
    
    # 4. 策略优化
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> RLOO 在理论上更优,但需要更多样本(建议每个 prompt 生成 8 个以上样本)。
