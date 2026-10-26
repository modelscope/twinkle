# Copyright (c) ModelScope Contributors. All rights reserved.
"""HCCL (Ascend NPU) transport for the delta pipeline (the pipeline itself lives in :mod:`.pipeline`).

Group membership: trainer rank 0 is group rank 0 and the vLLM worker processes
themselves are group ranks 1..N. There are no rollout-side CheckpointEngine
workers on NPU, so this engine only SENDS; the receiver is a vLLM-side component.

Lazy communicator
-----------------
``init_process_group`` only records rank, world size and the rendezvous metadata.
Trainer rank 0 builds the PyHccl communicator inside ``_ensure_wire()``, i.e. at
the first publish. The group completes only after every receiver has joined, and
every receiver connects its zmq SUB *before* joining the group, so "communicator
ready" implies "all receivers subscribed". A zmq PUB drops messages sent before a
subscriber has connected, so publishing the first manifest right after
``_ensure_wire()`` is lossless. The communicator is persistent across sync rounds
(HCCL group setup is expensive) and destroyed only in ``close()``.

Wire order per flush: manifest (zmq) -> positions broadcast (absent for values-only
flushes) -> values broadcast (HCCL, src=0).

All ``torch.npu`` / ``vllm_ascend`` usage is inside methods, so this module imports
on machines without torch_npu.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from twinkle.utils import find_free_port, find_node_ip, stateless_init_process_group
from twinkle.utils.platforms import hccl_socket_env
from ...base import CheckpointEngine
from ...hccl_checkpoint_engine import MasterMetadata
from ...zmq_pubsub import start_publisher
from .pipeline import DeltaSyncMixin

logger = logging.getLogger(__name__)


class DeltaHCCLCheckpointEngine(DeltaSyncMixin, CheckpointEngine):
    """Sparse delta weight sync over HCCL; trainer (sender) side only."""

    wire_format = 'delta_flush'

    def __init__(
        self,
        *args,
        bucket_size: int = 3072 << 20,
        group_name: str = 'twinkle_ckpt',
        rebuild_group: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        if rebuild_group:
            raise NotImplementedError('delta_sharded on NPU keeps one PyHccl broadcast group across sync rounds; '
                                      'rebuild_group=True is not supported')
        super().__init__(*args, **kwargs)
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype

        self.is_master = False
        self.topic = 'bucket_metadata'
        self.rank: int | None = None
        self.world_size: int | None = None
        self.socket = None

        self._comm = None
        self._rendezvous: MasterMetadata | None = None
        self._group_world_size: int | None = None
        self.ip: str | None = None
        self.zmq_port: int | None = None
        self.dist_port: int | None = None
        # HCCL socket env of trainer rank 0, published so vLLM workers join the group with the same ports.
        self.hccl_env: dict[str, str] | None = None

    # ---- lifecycle ----
    def prepare(self) -> MasterMetadata | None:
        """Master: bind the manifest PUB socket and allocate ports (idempotent). Others: None."""
        if not self.is_master:
            return None
        if self.socket is None:
            ip = find_node_ip()
            if not ip:
                from twinkle.infra import node_ip
                ip = node_ip()
            self.ip = ip
            self.zmq_port = find_free_port()
            self.dist_port = find_free_port()
            self.socket = start_publisher(self.ip, self.zmq_port)
            # The FSDP process group may already have pinned these from MASTER_PORT; workers must match.
            self.hccl_env = hccl_socket_env(self.dist_port)
        return MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Record only; the communicator is built lazily by rank 0 (see module docstring)."""
        self.rank = rank
        self.world_size = world_size
        if rank == 0:
            self._group_world_size = world_size
            self._rendezvous = master_metadata

    def _ensure_communicator(self) -> None:
        if self._comm is not None:
            return
        assert self.is_master and self.rank == 0 and self._rendezvous is not None, (
            'HCCL communicator is built by trainer rank 0 after init_process_group')
        self._comm = stateless_init_process_group(
            master_address=self._rendezvous.dist_ip,
            master_port=self._rendezvous.dist_port,
            rank=0,
            world_size=self._group_world_size,
            device=torch.npu.current_device(),
            backend='hccl',
        )
        logger.info('delta HCCL communicator ready: world_size=%s', self._group_world_size)

    def close(self):
        """Destroy the communicator, close the PUB socket and reset state. Idempotent."""
        comm, self._comm = self._comm, None
        if comm is not None:
            try:
                comm.destroyComm(comm.comm)
            except Exception as e:
                logger.warning('Error destroying HCCL communicator: %s', e)
        socket, self.socket = self.socket, None
        if socket is not None:
            ctx = getattr(socket, 'context', None)
            try:
                socket.close(linger=0)
            except Exception as e:
                logger.warning('Error closing ZMQ socket: %s', e)
            if ctx is not None:
                try:
                    ctx.term()
                except Exception as e:
                    logger.warning('Error terminating ZMQ context: %s', e)
        self.rank = None
        self.world_size = None
        self._rendezvous = None
        self._group_world_size = None
        self.hccl_env = None
        self.ip = self.zmq_port = self.dist_port = None
        self._shard_seeded = False

    def finalize(self):
        """Per-round finalize is a no-op: the group persists until ``close()``."""

    # ---- transport hooks ----
    @property
    def _wire_device(self):
        return torch.device('npu', torch.npu.current_device())

    def _ensure_wire(self) -> None:
        self._ensure_communicator()

    def _broadcast_flush(self, pos_u8, val_u8) -> None:
        comm = self._comm
        if pos_u8 is not None:
            comm.broadcast(pos_u8, src=0)
        comm.broadcast(val_u8, src=0)
        # PyHccl enqueues on the current stream and the caller frees the buffers right after.
        torch.npu.synchronize()

    def _release_after_round(self, phase: str) -> None:
        torch.npu.empty_cache()
        logger.debug('delta HCCL %s round done, NPU cache released', phase)

    # ---- not provided on NPU ----
    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]) -> Any:
        raise NotImplementedError('the delta manager builds its own topology; vLLM workers are the group receivers')

    async def receive_weights(self, *args, **kwargs):
        raise NotImplementedError('there is no rollout-side checkpoint engine on NPU; vLLM workers receive directly')
