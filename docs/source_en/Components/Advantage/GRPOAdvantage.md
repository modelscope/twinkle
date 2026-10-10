# GRPOAdvantage

GRPO (Group Relative Policy Optimization) advantage function calculates advantages by subtracting the group mean.

## Usage Example

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# Assume 2 prompts, each generating 4 samples
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 reward values
advantages = advantage_fn(rewards, num_generations=4, scale='none')

# Advantages will be each group minus the group mean:
# Group 1: [0.0-0.5, 1.0-0.5, 0.0-0.5, 1.0-0.5] = [-0.5, 0.5, -0.5, 0.5]
# Group 2: [1.0-0.25, 0.0-0.25, 0.0-0.25, 0.0-0.25] = [0.75, -0.25, -0.25, -0.25]
```

## How It Works

GRPO groups samples (each group corresponds to multiple generations from one prompt), then within each group:
1. Calculate the group mean reward
2. Advantage for each sample = reward - group mean
3. Optionally normalize the advantage values

This method:
- Reduces variance and improves training stability
- Performs relative comparisons within groups, better aligned with relative nature of human preferences
- Avoids the impact of reward scale

## Training Integration

Set `SamplingParams(num_samples=N, logprobs=1)` and read completions from each response's `sequences`. Training also requires aligned labels and old policy logprobs; advantages alone are insufficient.

See the maintained [GRPO training example](https://github.com/modelscope/twinkle/blob/main/cookbook/rl/grpo/short_math_grpo.py) for the complete sampling, reward, and optimizer loop. To use RLOO, replace the advantage function with `RLOOAdvantage` and generate at least two samples per prompt.
