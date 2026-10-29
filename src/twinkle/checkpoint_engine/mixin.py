# Copyright (c) ModelScope Contributors. All rights reserved.

from twinkle import Platform, remote_function
from twinkle.checkpoint_engine.base import CheckpointEngine


class CheckpointEngineMixin:

    _checkpoint_engine: CheckpointEngine = None
    _bucket_size: int = 3072 << 20  # 2 GB

    def _get_or_create_checkpoint_engine(self) -> 'CheckpointEngine':
        """Get or create the checkpoint engine instance (lazy singleton)."""
        if self._checkpoint_engine is None:
            engine_cls, engine_kwargs = Platform.get_platform().checkpoint_engine()
            self._checkpoint_engine = engine_cls(self._bucket_size, **engine_kwargs)
        return self._checkpoint_engine

    @remote_function(collect='first', lazy_collect=False)
    def prepare_checkpoint_engine(self, is_master):
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = is_master
        return engine.prepare()

    @remote_function(dispatch='slice', lazy_collect=True)
    def init_checkpoint_process_group(self, rank: int, world_size: int, master_metadata):
        """Initialize process group for weight synchronization."""
        if isinstance(rank, list):
            assert len(rank) == 1
            rank = rank[0]
        if isinstance(world_size, list):
            assert len(world_size) == 1
            world_size = world_size[0]
        if isinstance(master_metadata, list):
            assert len(master_metadata) == 1
            master_metadata = master_metadata[0]
        engine = self._get_or_create_checkpoint_engine()
        engine.init_process_group(
            rank=rank,
            world_size=world_size,
            master_metadata=master_metadata,
        )

    @remote_function(dispatch='all', lazy_collect=False)
    def finalize_checkpoint_engine(self):
        """Finalize checkpoint engine: release buffers, optionally destroy group."""
        if self._checkpoint_engine is not None:
            self._checkpoint_engine.finalize()
