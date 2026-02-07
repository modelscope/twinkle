# Copyright (c) ModelScope Contributors. All rights reserved.
"""CheckpointEngineMixin — shared checkpoint engine lifecycle for Model/Sampler.

Provides lazy-initialized checkpoint engine with prepare / init_process_group /
finalize methods.  Mixed into ``TransformersModel``, ``MegatronModel``, and
``VLLMSampler`` so that the common boilerplate is written only once.

Only activated when ``CheckpointEngineManager`` calls these methods via
``actor.method.remote()``.  When weight sync is not used, the engine is
never created and has zero overhead.
"""

import logging

import torch

from twinkle import remote_function

from twinkle.checkpoint_engine.base import CheckpointEngine

logger = logging.getLogger(__name__)


class CheckpointEngineMixin:
    """Mixin that adds checkpoint engine lifecycle to Model/Sampler classes.

    Subclasses only need to implement the transport-specific method:
    - ``send_weights_via_checkpoint_engine`` (model side)
    - ``receive_weights_via_checkpoint_engine`` (sampler side)
    """

    _checkpoint_engine: "CheckpointEngine | None" = None
    _checkpoint_engine_backend: str = 'nccl'
    _checkpoint_engine_bucket_size: int = 2048 << 20  # 2 GB

    def _get_or_create_checkpoint_engine(self) -> "CheckpointEngine":
        """Get or create the checkpoint engine instance (lazy singleton)."""
        if self._checkpoint_engine is None:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                backend = 'hccl'
            else:
                backend = self._checkpoint_engine_backend
            from twinkle.checkpoint_engine import CheckpointEngineRegistry
            self._checkpoint_engine = CheckpointEngineRegistry.new(
                backend,
                bucket_size=self._checkpoint_engine_bucket_size,
            )
        return self._checkpoint_engine

    @remote_function(dispatch='all')
    def prepare_checkpoint_engine(self, is_master: bool = False):
        """Prepare checkpoint engine and return metadata for process group setup.

        The ``CheckpointEngineManager`` calls this with ``is_master=True`` for
        model actor[0] and ``is_master=False`` for all others.

        Args:
            is_master: Whether this worker is the broadcast source.
        """
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = is_master
        return engine.prepare()

    @remote_function(dispatch='all')
    def init_checkpoint_process_group(self, rank: int, world_size: int, master_metadata):
        """Initialize process group for weight synchronization."""
        engine = self._get_or_create_checkpoint_engine()
        engine.init_process_group(
            rank=rank,
            world_size=world_size,
            master_metadata=master_metadata,
        )

    @remote_function(dispatch='all')
    def finalize_checkpoint_engine(self):
        """Finalize checkpoint engine: release buffers, optionally destroy group."""
        if self._checkpoint_engine is not None:
            self._checkpoint_engine.finalize()
