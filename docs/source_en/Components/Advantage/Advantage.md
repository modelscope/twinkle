# Advantage

Advantage functions are components in reinforcement learning used to calculate the advantage of an action relative to the average performance. In RLHF training, advantage functions guide policy optimization.

## Basic Interface

```python
class Advantage:
    
    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        Calculate advantage values
        
        Args:
            rewards: List or tensor of reward values
            num_generations: Number of samples generated per prompt
            scale: Normalization method
                - 'group': Normalize per group (GRPO)
                - 'batch': Normalize across entire batch
                - 'none': No normalization
        
        Returns:
            Advantage tensor
        """
        ...
```

## Available Advantage Functions

Twinkle provides two advantage function implementations:

### GRPOAdvantage

GRPO (Group Relative Policy Optimization) advantage function calculates advantages by subtracting the group mean.

- Simple and efficient, suitable for most scenarios
- Reduces variance and improves training stability
- Performs relative comparisons within groups

See: [GRPOAdvantage](GRPOAdvantage.md)

### RLOOAdvantage

RLOO (Reinforcement Learning with Leave-One-Out) advantage function uses leave-one-out method to calculate baselines.

- Theoretically superior, reduces bias
- Requires more samples (recommend 8 or more)
- More accurate counterfactual baseline estimation

See: [RLOOAdvantage](RLOOAdvantage.md)

## How to Choose

- **GRPO**: Suitable for scenarios with fewer samples (around 4), high computational efficiency
- **RLOO**: Suitable for scenarios with more samples (8 or more), better theoretical performance

> The choice of advantage function has a significant impact on RLHF training effectiveness. It's recommended to choose based on computational resources and sample quantity.
