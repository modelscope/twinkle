# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint Engine for weight synchronization between trainer and rollout.

``CheckpointEngineManager`` supports three synchronization modes: direct
generator streaming for local objects (``naive``), CUDA IPC for colocated Ray
actors (``colocate``), and NCCL/HCCL for disaggregated Ray actors
(``standalone``).

Reference: https://github.com/volcengine/verl/tree/main/verl/checkpoint_engine

Usage:
    >>> from twinkle.checkpoint_engine import CheckpointEngineManager
    >>>
    >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
    >>> manager.sync_weights()  # blocking call
"""

from .base import CheckpointEngine, TensorMeta
from .hccl_checkpoint_engine import HCCLCheckpointEngine
from .ipc_checkpoint_engine import IPCCheckpointEngine
from .manager import CheckpointEngineManager, CheckpointEngineMode
from .mixin import CheckpointEngineMixin
# Import backend implementations to register them
from .nccl_checkpoint_engine import NCCLCheckpointEngine

__all__ = [
    'CheckpointEngine',
    'CheckpointEngineMixin',
    'CheckpointEngineManager',
    'CheckpointEngineMode',
    'NCCLCheckpointEngine',
    'HCCLCheckpointEngine',
    'IPCCheckpointEngine',
    'TensorMeta',
]
