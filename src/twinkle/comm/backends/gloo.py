# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gloo backend: CPU tensors over TCP. Serves CPU-only hosts and the backend-neutral test suite."""
from __future__ import annotations
from datetime import timedelta

import torch
import torch.distributed as dist

from ..base import Communicator, ReduceOp, Rendezvous, Work

_DEFAULT_TIMEOUT = timedelta(seconds=300)


def _torch_reduce_op(op: ReduceOp | str) -> dist.ReduceOp:
    return getattr(dist.ReduceOp, ReduceOp(op).name)


class GlooCommunicator(Communicator):
    """A ``ProcessGroupGloo`` built over its own ``TCPStore``, outside the default world."""

    backend_name = 'gloo'

    def __init__(self, rank: int, world_size: int, device: torch.device | str, store: dist.Store,
                 pg: dist.ProcessGroup) -> None:
        super().__init__(rank, world_size, device)
        if self.device.type != 'cpu':
            raise ValueError(f'gloo communicator only carries CPU tensors, got device {self.device}')
        self._store = store
        self._pg = pg

    @classmethod
    def create(cls, rendezvous: Rendezvous, rank: int, world_size: int, device: torch.device | str = 'cpu',
               *, timeout: timedelta = _DEFAULT_TIMEOUT, **options) -> GlooCommunicator:
        store = dist.TCPStore(
            host_name=rendezvous.host,
            port=rendezvous.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
            wait_for_workers=True,
        )
        pg = dist.ProcessGroupGloo(store, rank, world_size, timeout)
        return cls(rank, world_size, device, store, pg)

    def _check_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != 'cpu':
            raise ValueError(f'gloo communicator only carries CPU tensors, got {tensor.device}')

    def broadcast(self, tensor: torch.Tensor, src: int = 0, async_op: bool = False) -> Work | None:
        self._check_alive()
        self._check_rank(src)
        self._check_tensor(tensor)
        opts = dist.BroadcastOptions()
        opts.rootRank = src
        work = self._pg.broadcast([tensor], opts)
        if async_op:
            return work
        work.wait()
        return None

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp | str = ReduceOp.SUM,
                   async_op: bool = False) -> Work | None:
        self._check_alive()
        self._check_tensor(tensor)
        opts = dist.AllreduceOptions()
        opts.reduceOp = _torch_reduce_op(op)
        work = self._pg.allreduce([tensor], opts)
        if async_op:
            return work
        work.wait()
        return None

    def _destroy(self) -> None:
        # The group is not registered with the default world, so destroy_process_group does not
        # apply; dropping the last references releases the Gloo context and the store.
        self._pg = None
        self._store = None
