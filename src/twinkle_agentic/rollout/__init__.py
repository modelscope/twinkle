# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Rollout
from .bridge import extend_with_bridge
from .factory import build_rollout
from .multi_turn import MultiTurnRollout

__all__ = [
    'APIMultiTurnRollout',
    'MultiTurnRollout',
    'Rollout',
    'build_rollout',
    'extend_with_bridge',
]


def __getattr__(name: str):
    if name == 'APIMultiTurnRollout':
        from .api_multi_turn import APIMultiTurnRollout
        return APIMultiTurnRollout
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
