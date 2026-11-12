# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from .base import Advantage


def _check_unit(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return value


@remote_class(execute='first')
class GAEAdvantage(Advantage):
    """Token-level generalized advantage estimation over ragged per-sample response tokens.

    The inputs are per-token rows, so a caller holding only references builds it with
    ``remote_group`` and the rows are read where they live.
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95, normalize: bool = True):
        self.gamma = _check_unit('gamma', gamma)
        self.gae_lambda = _check_unit('gae_lambda', gae_lambda)
        self.normalize = normalize

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(
        self,
        rewards: Sequence[Any],
        values: Sequence[Sequence[float]],
        *,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """``(advantages, returns)``, one row per sample, each as long as its ``values`` row.

        ``values[i]`` is sample i's per-token value estimates; ``rewards[i]`` is its per-token
        rewards (the same length) or one scalar terminal reward, placed on its last token.
        ``gamma`` / ``gae_lambda`` / ``normalize`` override the constructor's for this call;
        ``normalize`` whitens the advantages over every response token of the batch.
        """
        import torch

        gamma = self.gamma if gamma is None else _check_unit('gamma', float(gamma))
        gae_lambda = self.gae_lambda if gae_lambda is None else _check_unit('gae_lambda', float(gae_lambda))
        should_normalize = self.normalize if normalize is None else normalize
        if len(rewards) != len(values):
            raise ValueError(f'{len(rewards)} reward rows for {len(values)} value rows.')
        reward_rows: List[List[float]] = []
        value_rows: List[List[float]] = []
        for index, (reward, value) in enumerate(zip(rewards, values)):
            value = [float(v) for v in value]
            if isinstance(reward, (list, tuple)):
                reward = [float(r) for r in reward]
                if len(reward) != len(value):
                    raise ValueError(f'row {index}: {len(reward)} token rewards vs {len(value)} values; '
                                     'the value lane must cover exactly the response tokens.')
            else:
                if not value:
                    raise ValueError(f'row {index} has no response token to carry its terminal reward.')
                reward = [0.0] * (len(value) - 1) + [float(reward)]
            reward_rows.append(reward)
            value_rows.append(value)
        lengths = [len(row) for row in value_rows]
        max_len = max(lengths, default=0)
        rewards_t = torch.tensor([row + [0.0] * (max_len - len(row)) for row in reward_rows], dtype=torch.float32)
        values_t = torch.tensor([row + [0.0] * (max_len - len(row)) for row in value_rows], dtype=torch.float32)
        masks = torch.tensor([[True] * length + [False] * (max_len - length) for length in lengths], dtype=torch.bool)

        advantages = torch.zeros_like(rewards_t)
        for batch_idx, length in enumerate(lengths):
            last_gae = rewards_t.new_zeros(())
            for pos in range(length - 1, -1, -1):
                next_value = values_t[batch_idx, pos + 1] if pos + 1 < length else rewards_t.new_zeros(())
                delta = rewards_t[batch_idx, pos] + gamma * next_value - values_t[batch_idx, pos]
                last_gae = delta + gamma * gae_lambda * last_gae
                advantages[batch_idx, pos] = last_gae

        returns = advantages + values_t
        if should_normalize:
            valid = advantages[masks]
            if valid.numel() > 1:
                advantages = torch.where(masks, (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8),
                                         advantages)
        return ([advantages[i, :length].tolist() for i, length in enumerate(lengths)],
                [returns[i, :length].tolist() for i, length in enumerate(lengths)])
