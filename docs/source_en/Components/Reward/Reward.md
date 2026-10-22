# Reward

Reward functions are components in RLHF training used to evaluate the quality of model outputs. They calculate reward scores based on model-generated trajectories to guide policy learning.

## Basic Interface

```python
class Reward:

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        """
        Calculate reward values

        Args:
            trajectories: List of model-generated trajectories
            ground_truths: List of ground truth trajectories

        Returns:
            List of reward values
        """
        ...
```

## MathReward

The math reward function evaluates the correctness of answers to mathematical problems.

```python
from twinkle.reward import MathReward

reward_fn = MathReward()
rewards = reward_fn(generated_trajectories, ground_truth_trajectories)
# rewards: List[float], 1.0 for correct, 0.0 for incorrect
```

## FormatReward

The format reward function checks whether the output conforms to a specified format.

```python
from twinkle.reward import FormatReward

reward_fn = FormatReward()
rewards = reward_fn(trajectories, ground_truths)
```

## Custom Reward Functions

You can create custom rewards by inheriting from the Reward base class or using functions:

```python
from twinkle.reward import Reward
from twinkle.data_format import Trajectory
from typing import List

class CustomReward(Reward):

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        for traj, gt in zip(trajectories, ground_truths):
            # Custom evaluation logic
            score = self._evaluate(traj, gt)
            rewards.append(score)
        return rewards

    def _evaluate(self, traj, gt):
        # Implement specific evaluation logic
        ...
```

Or using a function:

```python
def my_reward(trajectories, ground_truths):
    return [1.0 if t == gt else 0.0 for t, gt in zip(trajectories, ground_truths)]

# Use in training
rewards = my_reward(generated, ground_truths)
```

## Training Integration

The sampler returns a list of `SampleResponse` objects. Read generated token IDs from `response.sequences`, decode them, and construct `Trajectory` objects before calling a reward function that accepts trajectories. Repeat each prompt's ground truth for its generated sequences so rewards and advantages stay in the same order.

See the maintained [GRPO training example](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/short_math_grpo.py) for the complete workflow.
