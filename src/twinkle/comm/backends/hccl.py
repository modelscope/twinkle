# Copyright (c) ModelScope Contributors. All rights reserved.
"""HCCL backend: a ``ProcessGroupHCCL`` over its own ``TCPStore``, outside the default world.

Serves Ascend NPU devices. One rank per device: HCCL refuses two ranks of one group on the
same device.

HCCL derives its internal listen/connect ports from a base port, so every rank of a group
must agree on ``HCCL_IF_BASE_PORT`` and the host/NPU socket port ranges. Ranks that already
pinned a different layout (a trainer whose default process group derived it from
``MASTER_PORT``) time out in the group's first collective, so a rank that knows the layout
(usually rank 0) publishes it and every rank passes it as ``socket_env``. Without it a rank
applies the layout derived from the rendezvous port *only if it has not pinned one already*,
so a process that pinned a layout earlier would silently disagree with its peers: pass
``socket_env`` whenever any rank may have pinned one.
"""
from __future__ import annotations
import uuid
from datetime import timedelta

import torch
import torch.distributed as dist

from ..base import Communicator, ReduceOp, Rendezvous, Work, normalize_device

#: TCPStore key rank 0 publishes this group's communicator name under.
_GROUP_ID_KEY = '__twinkle_hccl_group_id__'


def _group_id(rendezvous: Rendezvous, rank: int, store: dist.Store) -> str:
    """A communicator name that agrees across the group's ranks and is unique per group.

    HCCL names its communicator after ``Options.group_id``, and the default id is empty for
    everyone: a second live group in one process therefore fails ``hcclCommInitRootInfoConfig``
    with a parameter error. Deriving the name from the rendezvous address is not enough, because
    the API only requires the ranks to agree on the *address*, not on how they spell it
    (``0.0.0.0``/``localhost``/IPv6 long forms), and an unbounded ``group_name`` can exceed HCCL's
    name limit. Rank 0 mints the token and the peers read it from the store, so the name is
    identical by construction and bounded.
    """
    if rank == 0:
        store.set(_GROUP_ID_KEY, uuid.uuid4().hex)
    token = store.get(_GROUP_ID_KEY).decode()
    return f'{rendezvous.group_name}-{token[:16]}'


def _bind_layout(pg: dist.ProcessGroup, device: torch.device) -> None:
    """Run one tiny collective so the HCCL communicator is really built.

    HCCL builds its communicator -- and reads the socket layout environment -- on the group's
    first collective, not in the constructor. Waiting for one collective here is what lets
    :func:`hccl_socket_env_scope` restore the process environment as soon as ``_create`` returns.
    """
    from twinkle.utils.framework import Torch

    probe = torch.zeros(1, device=device)
    options = dist.AllreduceOptions()
    options.reduceOp = dist.ReduceOp.SUM
    pg.allreduce([probe], options).wait()
    Torch.synchronize(device)


class HCCLCommunicator(Communicator):
    """Device collectives through HCCL. The HCCL communicator is built by the first collective,
    which ``_create`` runs itself so the socket layout can be scoped to it."""

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

        from twinkle.utils.platforms import hccl_socket_env_scope

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
        options.group_id = _group_id(rendezvous, rank, store)
        # The layout is applied only for as long as it takes to bind it: leaving it in place would
        # hand this group's ports to the next group built in this process.
        with hccl_socket_env_scope(socket_env, master_port=rendezvous.port):
            pg = ProcessGroupHCCL(dist.PrefixStore(rendezvous.group_name, store), rank, world_size, options)
            _bind_layout(pg, device)
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
