# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL backend: a ``ProcessGroupNCCL`` over its own ``TCPStore``, outside the default world.

Serves CUDA devices (NVIDIA GPU and CUDA-compatible platforms such as Metax). One rank per
device: NCCL refuses two ranks of one group on the same device.
"""
from __future__ import annotations
from datetime import timedelta

import torch
import torch.distributed as dist

from ..base import Communicator, ReduceOp, Rendezvous, Work


class NCCLCommunicator(Communicator):
    """Device collectives through NCCL. The NCCL communicator is built lazily on the first op."""

    backend_name = 'nccl'

    def __init__(self, rank: int, world_size: int, device: torch.device | str, store: dist.Store,
                 pg: dist.ProcessGroup) -> None:
        super().__init__(rank, world_size, device)
        self._store = store
        self._pg = pg

    @property
    def native(self) -> dist.ProcessGroup:
        return self._pg

    @classmethod
    def _create(cls, rendezvous: Rendezvous, rank: int, world_size: int, device: torch.device, *,
                timeout: timedelta) -> NCCLCommunicator:
        if device.type != 'cuda':
            raise ValueError(f'nccl communicator needs a cuda device, got {device}')
        if not dist.is_nccl_available():
            raise RuntimeError('this torch build has no NCCL support')
        store = dist.TCPStore(
            host_name=rendezvous.host,
            port=rendezvous.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
            wait_for_workers=True,
        )
        pg = dist.ProcessGroupNCCL(dist.PrefixStore(rendezvous.group_name, store), rank, world_size, timeout)
        return cls(rank, world_size, device, store, pg)

    @staticmethod
    def _complete(work: Work, async_op: bool) -> Work | None:
        if async_op:
            return work
        # NCCL's wait() orders the caller's current stream after the collective; it does not
        # block the host (that is what synchronize() is for).
        work.wait()
        return None

    def _broadcast(self, tensor: torch.Tensor, src: int, async_op: bool) -> Work | None:
        opts = dist.BroadcastOptions()
        opts.rootRank = src
        return self._complete(self._pg.broadcast([tensor], opts), async_op)

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp, async_op: bool) -> Work | None:
        opts = dist.AllreduceOptions()
        opts.reduceOp = getattr(dist.ReduceOp, op.name)
        return self._complete(self._pg.allreduce([tensor], opts), async_op)

    # No _barrier override: ProcessGroupNCCL.barrier() picks its device from the calling thread's
    # current device, which the contract forbids; the base all_reduce barrier uses self.device.

    def _destroy(self) -> None:
        # Let queued collectives drain before the NCCL communicator goes away; the group is not
        # registered with the default world, so destroy_process_group does not apply.
        self.synchronize()
        pg, self._pg, self._store = self._pg, None, None
        if pg is not None and hasattr(pg, 'shutdown'):
            pg.shutdown()
