# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/nccl_checkpoint_engine.py
"""NCCL-based checkpoint engine for disaggregated trainer and rollout.

This engine uses NCCL broadcast for efficient GPU-to-GPU weight transfer
across different processes/nodes. It supports:
- Double buffering for pipelined transfer
- ZMQ for metadata, NCCL for weight data
- Streaming weight transfer to avoid OOM
- Persistent resources: NCCL group, ZMQ sockets, and buffers are reused
  across multiple sync calls to avoid costly re-initialization (~4s per call).

This implementation uses torch.distributed ProcessGroupNCCL directly for
NCCL operations.  A dedicated TCPStore handles rendezvous between the
participating Ray actors, completely independent of any existing default
process group.  This avoids NCCL version conflicts between CuPy (compiled
against NCCL 2.25) and the runtime NCCL 2.27 loaded by PyTorch.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import torch
import torch.distributed as dist
import zmq
from twinkle import get_logger

from twinkle.utils.network import (
    find_free_port,
    is_valid_ipv6_address,
)
from .base import CheckpointEngine, TensorMeta

logger = get_logger()


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int
    # TCPStore address for the checkpoint NCCL process group
    nccl_store_host: str = ""
    nccl_store_port: int = 0


def _pg_broadcast(pg: dist.ProcessGroup, tensor: torch.Tensor, src: int = 0):
    """Broadcast *tensor* using a raw (unregistered) ProcessGroupNCCL.

    ``dist.broadcast()`` requires a *registered* process group.  Since we
    create the PG directly via ``ProcessGroupNCCL(store, rank, world_size)``
    (which is NOT registered with the default ``_World``), we fall back to
    the low-level C++ ``pg.broadcast([tensor], opts)`` API.
    """
    opts = dist.BroadcastOptions()
    opts.rootRank = src
    work = pg.broadcast([tensor], opts)
    work.wait()


class BroadcastOperation:
    """Async broadcast operation with NCCL in separate thread.

    Wraps ``ProcessGroupNCCL.broadcast`` to run asynchronously so the main
    thread can continue processing (e.g. filling the next bucket) while the
    current bucket is being broadcast.

    Args:
        rank: The rank of the current process.
        pg: The torch.distributed ProcessGroup (unregistered NCCL).
        bucket: The GPU tensor buffer to broadcast.
        metadata: The metadata of tensors in the bucket.
        socket: The ZMQ socket for metadata communication.
        topic: The ZMQ topic for pub/sub.
    """

    def __init__(
        self,
        rank: int,
        pg: dist.ProcessGroup,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.pg = pg
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # Broadcast tensor metadata via ZMQ PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()

        # Broadcast tensor data via NCCL
        _pg_broadcast(self.pg, self.bucket, src=0)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            The bucket metadata after broadcast.
        """
        await self._task
        return self.metadata


class NCCLCheckpointEngine(CheckpointEngine):

    def __init__(
        self,
        bucket_size: int = 2048 << 20,
        group_name: str = "twinkle_ckpt",
        rebuild_group: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype

        # Set by Manager before prepare() via attribute assignment
        self.is_master = False
        self.topic = "bucket_metadata"

        # Will be set during prepare / init_process_group
        self.rank = None
        self.world_size = None
        self.send_buf = None
        self.recv_buf = None
        self.socket = None

        # torch.distributed process group for checkpoint NCCL ops
        self._pg: dist.ProcessGroup | None = None
        self._store: dist.Store | None = None

        # Track whether resources are ready for reuse
        self._prepared = False
        self._group_initialized = False

    # ── ZMQ helpers ──────────────────────────────────────────────────────

    def _start_zmq_server(self):
        """Start ZMQ PUB server for metadata broadcast (master only)."""
        import ray
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port = find_free_port()

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        """Connect to the ZMQ PUB server as a subscriber (receiver only)."""
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    # ── Core lifecycle ───────────────────────────────────────────────────

    def prepare(self) -> MasterMetadata | None:
        """Allocate double buffers and start ZMQ server (master only).

        Idempotent: if buffers and ZMQ are already set up, returns cached
        metadata without re-allocating.

        Returns:
            MasterMetadata with ZMQ IP/port and TCPStore address if master,
            else None.
        """
        if self._prepared:
            # Already prepared — return cached metadata
            if self.is_master:
                return MasterMetadata(
                    zmq_ip=self.ip,
                    zmq_port=self.listen_port,
                    nccl_store_host=self._nccl_store_host,
                    nccl_store_port=self._nccl_store_port,
                )
            return None

        if self.is_master:
            # Buffers on CUDA for NCCL broadcast
            self.send_buf = torch.zeros(
                self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(
                self.bucket_size, dtype=torch.uint8, device="cuda")
            self._start_zmq_server()

            # Allocate a TCPStore port for the checkpoint process group
            self._nccl_store_host = self.ip
            self._nccl_store_port = find_free_port()

            self._prepared = True
            return MasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.listen_port,
                nccl_store_host=self._nccl_store_host,
                nccl_store_port=self._nccl_store_port,
            )
        else:
            self.send_buf = torch.zeros(
                self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(
                self.bucket_size, dtype=torch.uint8, device="cuda")
            self._prepared = True
            return None

    def finalize(self):
        """Clean up resources after a sync.

        When ``rebuild_group=False`` (default): keeps NCCL group, ZMQ sockets,
        and buffers alive for the next sync.

        When ``rebuild_group=True``: destroys NCCL group and ZMQ sockets,
        forces a full re-init on the next sync.
        """
        if self.rebuild_group:
            # Full teardown
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning(f"Error closing ZMQ socket: {e}")
                self.socket = None

            if self._pg is not None:
                # Release PG by dropping references; do NOT call
                # dist.destroy_process_group as the PG is unregistered.
                self._pg = None
                self._store = None

            self.rank = None
            self.world_size = None
            self.send_buf = None
            self.recv_buf = None
            self._prepared = False
            self._group_initialized = False

        # When rebuild_group=False: keep everything alive for next sync

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Build communication topology for NCCL broadcast.

        The topology assigns:
        - Trainer rank 0 -> broadcast source (NCCL rank 0)
        - Other trainer ranks -> rank -1 (not participating)
        - Rollout workers -> ranks 1, 2, 3, ... (receivers)

        Args:
            trainer_world_size: Number of trainer workers.
            rollout_world_size: Number of rollout workers.
            metadata: List of metadata from prepare() calls.
                      metadata[0] is the MasterMetadata from trainer rank 0.

        Returns:
            Tuple of (trainer_kwargs, rollout_kwargs) for init_process_group().
        """
        master_metadata = metadata[0]

        trainer_kwargs = {
            "rank": [0] + [-1] * (trainer_world_size - 1),
            "world_size": [rollout_world_size + 1] * trainer_world_size,
            "master_metadata": [master_metadata] * trainer_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [master_metadata] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int,
                           master_metadata: MasterMetadata):
        """Initialize a dedicated NCCL process group for weight synchronization.

        Creates a ``ProcessGroupNCCL`` directly (without registering it in the
        default ``_World``), using a ``TCPStore`` hosted by the master for
        rendezvous.  This is completely independent of any existing
        ``torch.distributed`` default process group.

        Idempotent: if the group is already initialized and ``rebuild_group``
        is False, this is a fast no-op.

        Args:
            rank: The rank of this worker (-1 for non-participating trainers).
            world_size: Total number of workers in the sync group.
            master_metadata: Metadata from the master for ZMQ and store
                connection.
        """
        # Non-participating trainer ranks: record rank and return
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        # Fast path: group already initialized, skip all setup
        if self._group_initialized and not self.rebuild_group:
            return

        if self._pg is None:
            self.rank = rank
            self.world_size = world_size

            # Create a dedicated TCPStore for this checkpoint group.
            # Rank 0 (master / trainer) is the store server; all others
            # are clients that connect to it.
            is_store_master = (rank == 0)
            self._store = dist.TCPStore(
                host_name=master_metadata.nccl_store_host,
                port=master_metadata.nccl_store_port,
                world_size=world_size,
                is_master=is_store_master,
                wait_for_workers=True,
            )

            # Create a ProcessGroupNCCL directly — this does NOT interfere
            # with the default process group or any existing torch.distributed
            # state.
            self._pg = dist.ProcessGroupNCCL(
                self._store, rank, world_size,
            )
        else:
            assert self.rank == rank, f"rank {rank} != self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} != self.world_size {self.world_size}"
            )

        # Receivers connect to master's ZMQ PUB server
        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        # Barrier via broadcast to ensure all workers are ready
        barrier_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
        _pg_broadcast(self._pg, barrier_tensor, src=0)
        torch.cuda.synchronize()

        self._group_initialized = True
        logger.info(
            f"init_process_group: rank={self.rank}, "
            f"world_size={self.world_size}"
        )

    # ── Send / Receive ───────────────────────────────────────────────────

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
    ):
        """Send model weights to rollout workers via NCCL broadcast.

        Uses double buffering: fill send_buf while the previous bucket
        is being broadcast, then swap buffers.

        Args:
            weights: A generator yielding (name, tensor) pairs.
        """
        assert self.rank is not None and self.rank <= 0, (
            "Trainer workers other than rank 0 should not send weights."
        )

        # Non-participating ranks: consume the generator without sending
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0

        for name, weight in weights:
            # Check if bucket is full
            if offset + weight.nbytes > self.bucket_size:
                torch.cuda.synchronize()

                # Wait for previous broadcast to finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    pg=self._pg,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self.socket,
                    topic=self.topic,
                )

                # Swap buffers
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large "
                f"for bucket ({self.bucket_size / 1e6:.1f} MB). "
                f"Increase bucket_size."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }

            # Copy weight to buffer (both buffers are on CUDA)
            send_buf[offset:offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes

        # Broadcast final bucket
        torch.cuda.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self._pg,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        logger.info(
            f"Rank {self.rank} send weights done, "
            f"time cost: {time.time() - start_time:.2f}s"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive model weights from trainer via NCCL broadcast.

        Uses double buffering: receive into recv_buf while processing
        send_buf, then swap.

        Yields:
            Tuples of (name, tensor) for each weight.  The tensor is a
            *view* into the receive buffer -- callers that need to keep it
            should clone it.
        """
        assert self.rank is not None and self.rank > 0, (
            "Rank 0 should not receive weights."
        )

        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # Receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self._pg,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # Swap buffers
        send_buf, recv_buf = recv_buf, send_buf

        while not metadata["is_last"]:
            # 1. Start receiving next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                pg=self._pg,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. Yield tensors from current buffer (send_buf)
            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[
                    meta["offset"]:meta["offset"] + size
                ].view(dtype=dtype).view(shape)
                yield name, tensor

            # 3. Wait for next bucket
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. Swap buffers
            torch.cuda.synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        # Yield tensors from final bucket
        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[
                meta["offset"]:meta["offset"] + size
            ].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024)
        logger.info(
            f"receive_weights done: rank={self.rank}, "
            f"params={total_params}, "
            f"time={elapsed:.2f}s, bandwidth={bandwidth:.2f} GB/s"
        )
