# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client training runs the local MultiTurnRollout over an :class:`HTTPSampler` and a local
StreamingDriver; :class:`ClientCheckpointManager` is its weight sync."""
from ._http_sampler import ClientCheckpointManager, HTTPSampler

__all__ = ['ClientCheckpointManager', 'HTTPSampler']
