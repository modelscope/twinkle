# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Advantage
from .grpo import GRPOAdvantage
from .rloo import RLOOAdvantage


# TODO: Temporary helpers added to unblock cookbook/grpo examples.
# Each call creates a new Advantage instance, not suitable for production.
# Remove once the framework provides a proper advantage computation API.
def compute_advantages(rewards, num_generations=1, scale='group', **kwargs):
    """Backward-compatible helper for GRPO advantage computation."""
    return GRPOAdvantage()(
        rewards=rewards,
        num_generations=num_generations,
        scale=scale,
        **kwargs,
    )


def compute_advantages_rloo(rewards, num_generations=1, scale='group', **kwargs):
    """Backward-compatible helper for RLOO advantage computation."""
    return RLOOAdvantage()(
        rewards=rewards,
        num_generations=num_generations,
        scale=scale,
        **kwargs,
    )


__all__ = [
    'Advantage',
    'GRPOAdvantage',
    'RLOOAdvantage',
    'compute_advantages',
    'compute_advantages_rloo',
]
