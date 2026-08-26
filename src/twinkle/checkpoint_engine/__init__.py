# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint Engine for weight synchronization between trainer and rollout.

In colocated/local (including torchrun) deployments,
``CheckpointEngineManager`` streams the model's weight generator directly to
the sampler and does not create a checkpoint engine.  When both components
are Ray actor handlers, it provides NCCL/HCCL-based weight broadcast between
training model workers and inference sampler workers in disaggregated mode.

Reference: https://github.com/volcengine/verl/tree/main/verl/checkpoint_engine

Usage:
    >>> from twinkle.checkpoint_engine import CheckpointEngineManager
    >>>
    >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
    >>> manager.sync_weights()  # blocking call
"""

from .base import CheckpointEngine, TensorMeta
from .hccl_checkpoint_engine import HCCLCheckpointEngine
from .manager import CheckpointEngineManager
from .mixin import CheckpointEngineMixin
# Import backend implementations to register them
from .nccl_checkpoint_engine import NCCLCheckpointEngine

__all__ = [
    'CheckpointEngine',
    'CheckpointEngineMixin',
    'CheckpointEngineManager',
    'NCCLCheckpointEngine',
    'HCCLCheckpointEngine',
    'TensorMeta',
]
