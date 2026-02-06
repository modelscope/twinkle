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

This implementation uses ray.util.collective for NCCL operations, which is
compatible with Ray's distributed execution model.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, Union
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import ray
import ray.util.collective as collective
import torch
import zmq

from twinkle.utils.network import (
    find_free_port,
    is_valid_ipv6_address,
)
from .base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta

logger = logging.getLogger(__name__)


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


class BroadcastOperation:
    """Async broadcast operation with NCCL in separate thread.

    Wraps NCCL broadcast to run asynchronously so the main thread can
    continue processing (e.g. filling the next bucket) while the current
    bucket is being broadcast.

    Args:
        rank: The rank of the current process.
        group_name: The name of the NCCL process group.
        bucket: The tensor buffer to broadcast (cupy or torch tensor).
        metadata: The metadata of tensors in the bucket.
        socket: The ZMQ socket for metadata communication.
        topic: The ZMQ topic for pub/sub.
    """

    def __init__(
        self,
        rank: int,
        group_name: str,
        bucket: Union[torch.Tensor, cp.ndarray],
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        self.rank = rank
        self.group_name = group_name
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

        # Broadcast tensor data via ray.util.collective
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        """Wait for the broadcast operation to complete.

        Returns:
            The bucket metadata after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):
    """NCCL checkpoint engine with collective communication.

    All heavy resources (NCCL group, ZMQ sockets, GPU buffers) are
    **persistent** — they are created once during the first ``prepare()`` /
    ``init_process_group()`` call and reused across subsequent syncs.
    ``finalize()`` only releases buffers by default; set ``rebuild_group=True``
    if you need to tear everything down each sync.

    Args:
        bucket_size: Bucket size in bytes for weight transfer.
            Note: Memory overhead is 2 * bucket_size due to double buffering.
        group_name: Name of the NCCL process group.
        rebuild_group: Whether to destroy the NCCL group after each sync.
        rollout_dtype: Target dtype for weights.
    """

    def __init__(
        self,
        bucket_size: int,
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

        # Track whether resources are ready for reuse
        self._prepared = False
        self._group_initialized = False

    # ── ZMQ helpers ──────────────────────────────────────────────────────

    def _start_zmq_server(self):
        """Start ZMQ PUB server for metadata broadcast (master only)."""
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
            MasterMetadata with ZMQ IP/port if master, else None.
        """
        if self._prepared:
            # Already prepared — return cached metadata
            if self.is_master:
                return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port)
            return None

        # Master uses cupy to avoid memory register error with
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        if self.is_master:
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self._start_zmq_server()
            self._prepared = True
            return MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
            self._prepared = True
            return None

    def finalize(self):
        """Clean up resources after a sync.

        When ``rebuild_group=False`` (default): keeps NCCL group, ZMQ sockets,
        and buffers alive for the next sync.  Only useful for explicit cleanup
        if you want to reclaim GPU memory between syncs.

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

            if self.rank is not None and self.rank >= 0:
                try:
                    collective.destroy_collective_group(self.group_name)
                except Exception as e:
                    logger.warning(f"Error destroying collective group: {e}")

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
        - Trainer rank 0 → broadcast source (NCCL rank 0)
        - Other trainer ranks → rank -1 (not participating)
        - Rollout workers → ranks 1, 2, 3, ... (receivers)

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

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize the NCCL process group for weight synchronization.

        Idempotent: if the group is already initialized and ``rebuild_group``
        is False, this is a fast no-op (skips NCCL group creation, ZMQ
        connection, and barrier).

        Args:
            rank: The rank of this worker (-1 for non-participating trainers).
            world_size: Total number of workers in the sync group.
            master_metadata: Metadata from the master for ZMQ connection.
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

        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} != self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} != self.world_size {self.world_size}"
            )

        # Receivers connect to master's ZMQ PUB server
        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        # Barrier to ensure all workers are ready
        collective.barrier(self.group_name)

        self._group_initialized = True
        logger.info(f"init_process_group: rank={self.rank}, world_size={self.world_size}")

    # ── Send / Receive ───────────────────────────────────────────────────

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
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
                    group_name=self.group_name,
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

            # Copy weight to buffer
            if isinstance(send_buf, cp.ndarray):
                send_buf[offset:offset + weight.nbytes] = cp.asarray(
                    weight.view(-1).view(torch.uint8)
                )
            else:
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
            group_name=self.group_name,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive model weights from trainer via NCCL broadcast.

        Uses double buffering: receive into recv_buf while processing
        send_buf, then swap.

        Yields:
            Tuples of (name, tensor) for each weight.  The tensor is a
            *view* into the receive buffer — callers that need to keep it
            should clone it.
        """
        assert self.rank is not None and self.rank > 0, "Rank 0 should not receive weights."

        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # Receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            group_name=self.group_name,
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
                group_name=self.group_name,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. Yield tensors from current buffer (send_buf)
            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta["offset"]:meta["offset"] + size].view(dtype=dtype).view(shape)
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
            tensor = send_buf[meta["offset"]:meta["offset"] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024)
        logger.info(
            f"receive_weights done: rank={self.rank}, params={total_params}, "
            f"time={elapsed:.2f}s, bandwidth={bandwidth:.2f} GB/s"
        )
