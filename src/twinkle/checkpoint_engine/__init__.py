# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint Engine for weight synchronization between trainer and rollout.

Provides NCCL/HCCL-based weight broadcast from training model workers to
inference sampler workers in STANDALONE (disaggregated) deployment mode.

Reference: https://github.com/volcengine/verl/tree/main/verl/checkpoint_engine

Usage:
    >>> from twinkle.checkpoint_engine import CheckpointEngineManager
    >>>
    >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
    >>> manager.sync_weights()  # blocking call
"""

from .base import (
    CheckpointEngine,
    TensorMeta,
)
from .manager import CheckpointEngineManager
from .mixin import CheckpointEngineMixin

# Import backend implementations to register them
from .nccl_checkpoint_engine import NCCLCheckpointEngine
from .hccl_checkpoint_engine import HCCLCheckpointEngine

__all__ = [
    "CheckpointEngine",
    "CheckpointEngineMixin",
    "CheckpointEngineManager",
    "NCCLCheckpointEngine",
    "HCCLCheckpointEngine",
    "TensorMeta",
]
