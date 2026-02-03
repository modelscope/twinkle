# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Literal, Union, List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch


class Advantage:

    def __call__(self,
                 rewards: Union['torch.Tensor', List[float]],
                 num_generations: int = 1,
                 scale: Literal['group', 'batch', 'none'] = 'group',
                 **kwargs) -> 'torch.Tensor':
        """
        Advantage computation functions for RL training.

        Provides two methods:
        - compute_advantages: GRPO-style (subtract group mean)
        - compute_advantages_rloo: RLOO-style (leave-one-out baseline)

        Example:
            >>> from twinkle.rl import GRPOAdvantage
            >>> rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # 2 prompts, 4 samples each
            >>> advantages = GRPOAdvantage()(rewards, num_generations=4)
        """
        ...