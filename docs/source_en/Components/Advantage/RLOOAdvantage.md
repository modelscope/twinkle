# RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) advantage function uses leave-one-out method to calculate baselines.

## Usage Example

```python
from twinkle.advantage import RLOOAdvantage

advantage_fn = RLOOAdvantage()

rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
advantages = advantage_fn(rewards, num_generations=4)

# For each sample, the baseline is the mean of all other samples
# First sample in first group: 0.0 - mean([1.0, 0.0, 1.0]) = 0.0 - 0.667 = -0.667
# ...
```

## How It Works

For each sample, RLOO:
1. Calculates the mean reward of all other samples in the group (leave-one-out baseline)
2. Advantage = sample reward - leave-one-out baseline
3. Optionally normalizes the values

RLOO advantages:
- Avoids using the sample's own information as baseline, reducing bias
- More accurate counterfactual baseline estimation
- Better performance when there are more samples

## Complete Training Example

```python
from twinkle.advantage import RLOOAdvantage
from twinkle.model import TransformersModel
from twinkle.sampler import vLLMSampler
from twinkle.reward import MathReward

# Create components
actor = TransformersModel(model_id='Qwen/Qwen2.5-7B-Instruct')
sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
reward_fn = MathReward()
advantage_fn = RLOOAdvantage()

# Training loop
for batch in dataloader:
    # 1. Sample generation (generate more samples to improve RLOO effectiveness)
    response = sampler.sample(batch, num_samples=8)
    
    # 2. Calculate rewards
    rewards = reward_fn(response.trajectories, batch.ground_truths)
    
    # 3. Calculate advantages
    advantages = advantage_fn(rewards, num_generations=8)
    
    # 4. Policy optimization
    loss = actor.forward_backward(
        inputs=response.inputs,
        advantages=advantages
    )
    actor.clip_grad_and_step()
```

> RLOO is theoretically superior but requires more samples (recommend 8 or more samples per prompt).
