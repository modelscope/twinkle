# Gym

Gym 组件为 Twinkle 中的强化学习环境提供接口。

```python
from twinkle.gym import Gym

class CustomGym(Gym):

    def step(self, trajectories, **kwargs):
        """
        执行一个 RL 步骤：评估轨迹并返回奖励。

        Args:
            trajectories: 模型生成的待评估轨迹
            **kwargs: 额外参数

        Returns:
            每个轨迹的奖励值
        """
        ...
```

Gym 抽象允许你插入自定义 RL 环境与训练循环交互。它将奖励计算和环境交互与核心训练逻辑解耦。

> Gym 通常用于在线策略 RL 训练中，环境需要对模型生成的输出提供反馈。
