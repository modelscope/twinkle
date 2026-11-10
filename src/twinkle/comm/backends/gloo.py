# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gloo backend: tensors over TCP through host memory.

Serves CPU-only hosts, platforms without a device collective library (MPS) and the
backend-neutral test suite. Non-CPU tensors are staged through a host copy.
"""
from __future__ import annotations
from datetime import timedelta

import torch
import torch.distributed as dist

from ..base import Communicator, ReduceOp, Rendezvous, Work


class _StagedWork:
    """Completes a host-staged operation: wait, then copy the host result back into the tensor."""

    def __init__(self, work: Work, tensor: torch.Tensor, host: torch.Tensor) -> None:
        self._work = work
        self._tensor = tensor
        self._host = host
        self._done = False

    def wait(self) -> bool:
        if not self._done:
            self._work.wait()
            self._tensor.copy_(self._host)
            self._done = True
        return True


class GlooCommunicator(Communicator):
    """A ``ProcessGroupGloo`` built over its own ``TCPStore``, outside the default world."""

    backend_name = 'gloo'

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
                timeout: timedelta) -> GlooCommunicator:
        store = dist.TCPStore(
            host_name=rendezvous.host,
            port=rendezvous.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
            wait_for_workers=True,
        )
        pg = dist.ProcessGroupGloo(dist.PrefixStore(rendezvous.group_name, store), rank, world_size, timeout)
        return cls(rank, world_size, device, store, pg)

    def _complete(self, work: Work, tensor: torch.Tensor, host: torch.Tensor, async_op: bool) -> Work | None:
        if host is not tensor:
            work = _StagedWork(work, tensor, host)
        if async_op:
            return work
        work.wait()
        return None

    def _broadcast(self, tensor: torch.Tensor, src: int, async_op: bool) -> Work | None:
        host = tensor if tensor.device.type == 'cpu' else tensor.cpu()
        opts = dist.BroadcastOptions()
        opts.rootRank = src
        return self._complete(self._pg.broadcast([host], opts), tensor, host, async_op)

    def _all_reduce(self, tensor: torch.Tensor, op: ReduceOp, async_op: bool) -> Work | None:
        host = tensor if tensor.device.type == 'cpu' else tensor.cpu()
        opts = dist.AllreduceOptions()
        opts.reduceOp = getattr(dist.ReduceOp, op.name)
        return self._complete(self._pg.allreduce([host], opts), tensor, host, async_op)

    def _barrier(self) -> None:
        self._pg.barrier(dist.BarrierOptions()).wait()

    def _destroy(self) -> None:
        # The group is not registered with the default world, so destroy_process_group does not
        # apply; shut the Gloo context down and drop the store.
        pg, self._pg, self._store = self._pg, None, None
        if pg is not None and hasattr(pg, 'shutdown'):
            pg.shutdown()
