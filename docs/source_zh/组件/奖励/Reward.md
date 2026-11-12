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

## 训练集成

采样器返回 `SampleResponse` 列表。先从 `response.sequences` 读取生成 token、解码并构造 `Trajectory`，再调用接受 trajectory 的奖励函数。每个 prompt 的标准答案需要按生成数量重复，保证奖励和优势值的顺序一致。

完整流程见 [GRPO 训练示例](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/short_math_grpo.py)。
