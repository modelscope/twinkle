# Copyright (c) ModelScope Contributors. All rights reserved.
"""HCCL backend: a ``ProcessGroupHCCL`` over its own ``TCPStore``, outside the default world.

Serves Ascend NPU devices. One rank per device: HCCL refuses two ranks of one group on the
same device.

HCCL derives its internal listen/connect ports from a base port, so every rank of a group
must agree on ``HCCL_IF_BASE_PORT`` and the host/NPU socket port ranges. Ranks that already
pinned a different layout (a trainer whose default process group derived it from
``MASTER_PORT``) time out in the group's first collective, so a rank that knows the layout
(usually rank 0) publishes it and every rank passes it as ``socket_env``; without it the
layout is derived from the rendezvous port, which is identical on every rank.
"""
from __future__ import annotations
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from ..base import Communicator, ReduceOp, Rendezvous, Work, normalize_device


class HCCLCommunicator(Communicator):
    """Device collectives through HCCL. The HCCL communicator is built lazily on the first op."""

    backend_name = 'hccl'
    #: ``socket_env``: HCCL socket env every rank of this group must share (see module docstring).
    supported_options = frozenset({'socket_env'})

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
                timeout: timedelta, socket_env: dict[str, str] | None = None) -> HCCLCommunicator:
        if device.type != 'npu':
            raise ValueError(f'hccl communicator needs an npu device, got {device}')
        try:
            # Registers torch.npu and the HCCL process group type.
            import torch_npu  # noqa: F401
            from torch_npu._C._distributed_c10d import ProcessGroupHCCL
        except ImportError as error:
            raise RuntimeError('hccl communicator needs torch_npu (Ascend PyTorch)') from error
        # 'npu' without an index could not be resolved before torch_npu was imported.
        device = normalize_device(device)

        from twinkle.utils.platforms import ensure_hccl_socket_env
        if socket_env:
            # The publisher's layout wins over this process's own values; agreement is what
            # matters and only the publisher knows whether its group already pinned them.
            os.environ.update(socket_env)
        else:
            ensure_hccl_socket_env(rendezvous.port)

        store = dist.TCPStore(
            host_name=rendezvous.host,
            port=rendezvous.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
            wait_for_workers=True,
        )
        options = ProcessGroupHCCL.Options()
        options._timeout = timeout
        options.is_high_priority_stream = False
        # HCCL names its communicator after the group id, and a second live group with the same
        # (default empty) id in one process fails hcclCommInitRootInfoConfig with a parameter
        # error. The rendezvous address is identical on every rank and unique per live group.
        options.group_id = f'{rendezvous.group_name}@{rendezvous.host}:{rendezvous.port}'
        pg = ProcessGroupHCCL(dist.PrefixStore(rendezvous.group_name, store), rank, world_size, options)
        return cls(rank, world_size, device, store, pg)

    @staticmethod
    def _complete(work: Work, async_op: bool) -> Work | None:
        if async_op:
            return work
        # HCCL's wait() orders the caller's current stream after the collective; it does not
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

    # No _barrier override: ProcessGroupHCCL.barrier() picks its device from the calling thread's
    # current device, which the contract forbids; the base all_reduce barrier uses self.device.

    def _destroy(self) -> None:
        # Let queued collectives drain before the HCCL communicator goes away; the group is not
        # registered with the default world, so destroy_process_group does not apply.
        self.synchronize()
        pg, self._pg, self._store = self._pg, None, None
        if pg is not None and hasattr(pg, 'shutdown'):
            pg.shutdown()
