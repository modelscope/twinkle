# Copyright (c) ModelScope Contributors. All rights reserved.
"""CUDA rollout receiver: a Ray actor colocated with one vLLM worker.

It joins the trainer's NCCL group, receives each delta flush (:meth:`DeltaShardedCheckpointEngine.receive_weights`)
and relays it to its vLLM worker over CUDA IPC. The NPU counterpart is :mod:`.hccl_receiver`, which runs inside
the vLLM worker itself because NPU has no device IPC.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import torch

from ..config import DeltaReceiverPlacement, DeltaSyncConfig, DeltaTransportEndpoint
from ..transport.nccl_engine import DeltaShardedCheckpointEngine

TensorFlush = tuple[list[tuple[str, torch.Tensor]], bool]


class CudaDeltaReceiver:
    """Checkpoint worker colocated with one vLLM worker"""

    def __init__(self, config: DeltaSyncConfig, endpoint: DeltaTransportEndpoint,
                 placement: DeltaReceiverPlacement) -> None:
        visible_device = placement.get('cuda_visible_device', placement['device_uuid'])
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
        print(f'[delta receiver rank={placement["rank"]}] '
              f'CUDA_VISIBLE_DEVICES={visible_device!r}, expected_uuid={placement["device_uuid"]}', flush=True)
        try:
            torch.cuda.set_device(0)
        except RuntimeError as error:
            raise RuntimeError(
                f'CUDA delta receiver initialization failed: CUDA_VISIBLE_DEVICES={visible_device!r}, '
                f'expected_uuid={placement["device_uuid"]}, '
                f'LD_PRELOAD={os.environ.get("LD_PRELOAD", "")!r}') from error
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if actual_uuid != placement['device_uuid']:
            raise ValueError(f'CUDA delta receiver GPU mismatch: expected {placement["device_uuid"]}, '
                             f'got {actual_uuid}, CUDA_VISIBLE_DEVICES={visible_device!r}')
        self.config = config
        self.zmq_handle = placement['zmq_handle']
        self.engine = DeltaShardedCheckpointEngine(bucket_size=config.bucket_bytes)
        self.engine.prepare()
        self.engine.init_process_group(placement['rank'], endpoint['world_size'], endpoint['master_metadata'])
        self._flushes: Iterator[TensorFlush] | None = None
        self._current: TensorFlush | None = None

    def ready(self) -> int:
        return self.engine.rank

    def next_flush(self) -> dict[str, bool] | None:
        if self._flushes is None:
            self._flushes = self.engine.receive_weights()
        self._current = next(self._flushes, None)
        if self._current is None:
            self._flushes = None
            return None
        return {'is_last': self._current[1]}

    def send_flush(self) -> None:
        from twinkle.sampler.vllm_sampler.delta.cuda_ipc import BucketedWeightSender

        tensors, _ = self._current
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=max(1, self.config.bucket_bytes >> 20), use_shm=False)
        try:
            asyncio.run(sender.async_send_weights(iter(tensors)))
        finally:
            self._current = None

    def close(self) -> None:
        self._current = None
        if self._flushes is not None:
            self._flushes.close()
            self._flushes = None
        self.engine.close()
