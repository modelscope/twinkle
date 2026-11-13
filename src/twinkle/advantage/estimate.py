# Copyright (c) ModelScope Contributors. All rights reserved.
"""One entry over the per-sequence estimators: pick by name, forward the estimator-specific inputs."""
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .base import Advantage
from .gpg import GPGAdvantage
from .grpo import GRPOAdvantage
from .grpo_passk import GRPOPassKAdvantage
from .opo import OPOAdvantage
from .reinforce_plus_plus import ReinforcePlusPlusAdvantage
from .reinforce_pp_baseline import ReinforcePPBaselineAdvantage
from .remax import ReMaxAdvantage
from .rloo import RLOOAdvantage

ESTIMATORS = {
    'grpo': GRPOAdvantage,
    'rloo': RLOOAdvantage,
    'reinforce_plus_plus': ReinforcePlusPlusAdvantage,
    'reinforce_plus_plus_baseline': ReinforcePPBaselineAdvantage,
    'remax': ReMaxAdvantage,
    'opo': OPOAdvantage,
    'gpg': GPGAdvantage,
    'grpo_passk': GRPOPassKAdvantage,
}
SCALINGS = ('group', 'batch', 'none', 'gdpo')


def compute_advantages(rewards: Union[torch.Tensor, Sequence[Any]],
                       num_generations: int,
                       *,
                       reward_weights: Optional[Sequence[float]] = None,
                       advantage_estimator: Union[str, type, Advantage] = 'grpo',
                       scale_rewards: str = 'group',
                       kl_in_reward: bool = False,
                       beta: float = 0.0,
                       kl_values: Optional[Union[torch.Tensor, Sequence[float]]] = None,
                       reward_baselines: Optional[Union[torch.Tensor, Sequence[float]]] = None,
                       response_lengths: Optional[Union[torch.Tensor, Sequence[int]]] = None,
                       f_norm: float = 1.0) -> List[float]:
    """Per-sequence advantages from rewards, one per sample.

    ``rewards`` is a ``[N, n_funcs]`` matrix or a flat ``[N]``, each consecutive block of
    ``num_generations`` one prompt group; ``reward_weights`` (default all-ones) fold the columns.
    ``kl_in_reward`` subtracts ``beta * kl_values`` BEFORE normalization; ``remax`` needs
    ``reward_baselines``, ``opo`` ``response_lengths``, ``gpg`` ``f_norm``. The input is one scalar
    per sample, so this runs wherever the rewards are -- no remote placement needed.
    ``advantage_estimator`` is a registered name, an ``Advantage`` subclass of your own, or a
    ready instance -- the estimator subclasses are the algorithm extension point.
    """
    if isinstance(advantage_estimator, str):
        if advantage_estimator not in ESTIMATORS:
            raise ValueError(f'advantage_estimator {advantage_estimator!r} not in {tuple(ESTIMATORS)}.')
        estimator = ESTIMATORS[advantage_estimator]()
    elif isinstance(advantage_estimator, type):
        if not issubclass(advantage_estimator, Advantage):
            raise TypeError(f'advantage_estimator class {advantage_estimator!r} must subclass '
                            'twinkle.advantage.Advantage.')
        estimator = advantage_estimator()
    elif isinstance(advantage_estimator, Advantage):
        estimator = advantage_estimator
    else:
        raise TypeError('advantage_estimator must be a registered name, an Advantage subclass, or an '
                        'Advantage instance.')
    if scale_rewards not in SCALINGS:
        raise ValueError(f'scale_rewards {scale_rewards!r} not in {SCALINGS}.')
    matrix = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(list(rewards), dtype=torch.float32)
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(1)
    matrix = matrix.to(torch.float32)
    n, n_funcs = matrix.shape
    if num_generations <= 0 or n % num_generations != 0:
        raise ValueError(f'{n} rewards is not divisible by num_generations={num_generations}; '
                         'samples must be grouped as consecutive blocks of num_generations.')
    if kl_in_reward and kl_values is None:
        raise ValueError('kl_in_reward=True requires kl_values ([N] per-sample reference KL).')
    if reward_weights is None:
        weights = torch.ones(n_funcs, dtype=torch.float32)
    elif len(reward_weights) != n_funcs:
        raise ValueError(f'reward_weights length {len(reward_weights)} != number of reward funcs {n_funcs}.')
    else:
        weights = torch.tensor(list(reward_weights), dtype=torch.float32)
    # The estimators read only what they need (remax -> reward_baselines, opo -> response_lengths,
    # gpg -> f_norm) and ignore the rest.
    extra: Dict[str, Any] = {}
    if reward_baselines is not None:
        extra['reward_baselines'] = reward_baselines
    if response_lengths is not None and isinstance(estimator, OPOAdvantage):
        extra['response_lengths'] = response_lengths
    if isinstance(estimator, GPGAdvantage):
        extra['f_norm'] = f_norm
    advantages = estimator._per_sequence(
        matrix,
        num_generations,
        scale=scale_rewards,
        reward_weights=weights,
        kl_in_reward=kl_in_reward,
        beta=beta,
        kl_values=kl_values,
        **extra,
    )
    return advantages.tolist()
