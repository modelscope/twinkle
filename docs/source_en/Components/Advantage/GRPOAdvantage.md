# GRPOAdvantage

GRPO (Group Relative Policy Optimization) advantage function calculates advantages by subtracting the group mean.

## Usage Example

```python
from twinkle.advantage import GRPOAdvantage

advantage_fn = GRPOAdvantage()

# Assume 2 prompts, each generating 4 samples
rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 8 reward values
advantages = advantage_fn(rewards, num_generations=4, scale='group')

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

## Complete Training Example

Using the advantage function in GRPO training:

```python
from twinkle.advantage import GRPOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward

# Create components
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = GRPOAdvantage()

# Training loop
for batch in dataloader:
    # 1. Sample generation
    response = sampler.sample(batch, num_samples=4)
    
    # 2. Calculate rewards
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. Calculate advantages
    advantages = advantage_fn(rewards, num_generations=4)
    
    # 4. Policy optimization
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> The GRPO method is simple and efficient, suitable for most RLHF training scenarios.
