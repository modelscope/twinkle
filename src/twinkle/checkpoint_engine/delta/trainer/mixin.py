# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer-side Ray lifecycle for the delta checkpoint engine (NCCL on GPU, HCCL on NPU)."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from time import perf_counter
from typing import Any

import torch.distributed as dist

from twinkle import Platform, remote_function
from ..config import DeltaMasterMetadata, DeltaSyncConfig, DeltaTrainerMetadata, DeltaTransportEndpoint
from ..platform import DELTA_DEVICES, resolve_delta_engine_cls


class DeltaTrainerMixin:
    """Lifecycle RPCs only. Export hooks come from a DeltaTrainerBackend subclass on the same model."""
    _delta_engine: Any = None  # transport chosen by platform (see resolve_delta_engine_cls)
    _delta_metadata_group: dist.ProcessGroup | None = None
    _delta_master_metadata: DeltaMasterMetadata | None = None

    @staticmethod
    def _delta_rank() -> int:
        # A single trainer rank never initializes the default process group.
        return dist.get_rank() if dist.is_initialized() else 0

    def _validate_delta_ranks(self) -> None:
        error = None
        try:
            self.validate_delta_backend()
        except Exception as validation_error:
            error = str(validation_error)
        if self._delta_metadata_group is None:
            if error:
                raise ValueError(f'Unsupported delta configuration: {error}')
            return
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=self._delta_metadata_group)
        if any(errors):
            raise ValueError(f'Unsupported delta configuration: {errors}')

    def _initialize_delta_engine(self, config: DeltaSyncConfig) -> None:
        # Resolve first so an unsupported platform fails before any process group is created.
        platform_name = Platform.get_platform().__name__
        engine_cls = resolve_delta_engine_cls(platform_name)
        if dist.is_initialized():
            self._delta_metadata_group = dist.new_group(
                backend='gloo', timeout=timedelta(seconds=config.timeout_seconds))
        try:
            self._validate_delta_ranks()
        except Exception:
            if self._delta_metadata_group is not None:
                dist.destroy_process_group(self._delta_metadata_group)
                self._delta_metadata_group = None
            raise
        self._delta_device = DELTA_DEVICES[platform_name]
        self._delta_pin_snapshots = config.delta_pin_snapshots
        self._delta_engine = engine_cls(
            bucket_size=config.bucket_bytes, batch_gather=config.gather_batch_size)
        self._delta_engine.is_master = self._delta_rank() == 0
        self._delta_master_metadata = self._delta_engine.prepare()

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def prepare_delta_sync(self, config: DeltaSyncConfig) -> DeltaTrainerMetadata:
        self._lazy_wrap_model()
        if self._delta_engine is None:
            self._initialize_delta_engine(config)
        metadata: DeltaTrainerMetadata = {'device': self._delta_device, 'master_metadata': self._delta_master_metadata}
        hccl_env = getattr(self._delta_engine, 'hccl_env', None)
        if hccl_env:
            metadata['hccl_env'] = dict(hccl_env)
        return metadata

    @remote_function(dispatch='all', lazy_collect=True)
    def init_delta_transport(self, endpoint: DeltaTransportEndpoint) -> None:
        rank = 0 if self._delta_rank() == 0 else -1
        self._delta_engine.init_process_group(rank, endpoint['world_size'], endpoint['master_metadata'])

    @remote_function(dispatch='all', lazy_collect=True)
    def send_delta_weights(self) -> dict[str, Any]:
        started = perf_counter()
        metrics = asyncio.run(self._delta_engine.send_weights(self))
        return {'send_seconds': perf_counter() - started, 'checkpoint_engine': metrics}

    @remote_function(dispatch='all', lazy_collect=False)
    def close_delta_trainer(self) -> None:
        if self._delta_engine is not None:
            self._delta_engine.close()
            self._delta_engine = None
        snapshots = getattr(self, '_delta_shard_snap', {})
        snapshots.clear()
        if self._delta_metadata_group is not None:
            dist.destroy_process_group(self._delta_metadata_group)
            self._delta_metadata_group = None
