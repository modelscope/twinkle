# Reward

Reward (奖励函数) 是 RLHF 训练中用于评估模型输出质量的组件。奖励函数根据模型生成的轨迹计算奖励分数,用于指导策略学习。

## 基本接口

```python
class Reward:
    
    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        """
        计算奖励值
        
        Args:
            trajectories: 模型生成的轨迹列表
            ground_truths: 真实答案轨迹列表
        
        Returns:
            奖励值列表
        """
        ...
```

## MathReward

数学奖励函数用于评估数学问题的答案正确性。

```python
from twinkle.reward import MathReward

reward_fn = MathReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float],1.0 表示正确,0.0 表示错误
```

## FormatReward

格式奖励函数用于检查输出是否符合指定格式。

```python
from twinkle.reward import FormatReward

reward_fn = FormatReward()
rewards = reward_fn(trajectories, ground_truths)
```

## CountDownAccuracyReward

倒计时准确率奖励函数,在答案接近正确时给予部分奖励。

```python
from twinkle.reward import CountDownAccuracyReward

reward_fn = CountDownAccuracyReward()
rewards = reward_fn(trajectories, ground_truths)
```

## 自定义奖励函数

你可以通过继承 Reward 基类或使用函数来创建自定义奖励:

```python
from twinkle.reward import Reward
from twinkle.data_format import Trajectory
from typing import List

class CustomReward(Reward):
    
    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        for traj, gt in zip(trajectories, ground_truths):
            # 自定义评估逻辑
            score = self._evaluate(traj, gt)
            rewards.append(score)
        return rewards
    
    def _evaluate(self, traj, gt):
        # 实现具体评估逻辑
        ...
```

或使用函数:

```python
def my_reward(trajectories, ground_truths):
    return [1.0 if t == gt else 0.0 for t, gt in zip(trajectories, ground_truths)]

# 在训练中使用
rewards = my_reward(generated, ground_truths)
```

## 使用场景

奖励函数在 RLHF 训练的典型使用流程:

```python
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward
from twinkle.advantage import GRPOAdvantage

sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

for batch in dataloader:
    # 1. 采样生成多个候选答案
    response = sampler.sample(batch, num_samples=4)
    
    # 2. 使用奖励函数评估质量
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. 计算优势值
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. 用优势值进行策略梯度更新
    ...
```

> 奖励函数的设计对 RLHF 效果至关重要。好的奖励函数应该准确反映任务目标,并提供明确的学习信号。
