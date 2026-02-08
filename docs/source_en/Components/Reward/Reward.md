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

## CountDownAccuracyReward

The countdown accuracy reward function provides partial rewards when answers are close to correct.

```python
from twinkle.reward import CountDownAccuracyReward

reward_fn = CountDownAccuracyReward()
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

## Usage Scenarios

Typical workflow of reward functions in RLHF training:

```python
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward
from twinkle.advantage import GRPOAdvantage

sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

for batch in dataloader:
    # 1. Sample and generate multiple candidate answers
    response = sampler.sample(batch, num_samples=4)
    
    # 2. Evaluate quality using reward function
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. Calculate advantages
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. Update policy using advantage values
    ...
```

> The design of reward functions is crucial for RLHF effectiveness. A good reward function should accurately reflect the task objectives and provide clear learning signals.
