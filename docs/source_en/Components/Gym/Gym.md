# Gym

The Gym component provides an interface for reinforcement learning environments in Twinkle.

```python
from twinkle.gym import Gym

class CustomGym(Gym):

    def step(self, trajectories, **kwargs):
        """
        Execute one RL step: evaluate trajectories and return rewards.

        Args:
            trajectories: Model-generated trajectories to evaluate
            **kwargs: Additional arguments

        Returns:
            Reward values for each trajectory
        """
        ...
```

The Gym abstraction allows you to plug in custom RL environments that interact with the training loop. It decouples reward computation and environment interaction from the core training logic.

> Gym is typically used in on-policy RL training where the environment needs to provide feedback on model-generated outputs.
