# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle import remote_function, Platform
from twinkle.checkpoint_engine.base import CheckpointEngine


class CheckpointEngineMixin:

    _checkpoint_engine: "CheckpointEngine | None" = None
    _bucket_size: int = 2048 << 20  # 2 GB

    def _get_or_create_checkpoint_engine(self) -> "CheckpointEngine":
        """Get or create the checkpoint engine instance (lazy singleton)."""
        if self._checkpoint_engine is None:
            if Platform.get_platform().__name__ == 'GPU':
                from twinkle.checkpoint_engine import NCCLCheckpointEngine
                self._checkpoint_engine = NCCLCheckpointEngine(self._bucket_size)
            elif Platform.get_platform().__name__ == 'NPU':
                from twinkle.checkpoint_engine import HCCLCheckpointEngine
                self._checkpoint_engine = HCCLCheckpointEngine(self._bucket_size)
            self._checkpoint_engine.is_master = Platform.is_master()
            self._checkpoint_engine.prepare()
        return self._checkpoint_engine

    @remote_function(dispatch='all')
    def prepare_checkpoint_engine(self):
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = Platform.is_master()
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
