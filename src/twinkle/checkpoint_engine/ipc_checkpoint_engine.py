# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight sync between a trainer and a sampler that share a GPU (colocation).

The NCCL engine cannot be used for this. Two processes bound to the same device cannot form a NCCL
communicator at all -- it fails with ``Duplicate GPU detected : rank 1 and rank 0 both on CUDA
device``, so this is not a matter of picking the faster transport.

What replaces it is CUDA IPC: the sender packs weights into a device buffer and hands the receiver a
handle to it, and the receiver maps that same physical memory. Measured on one device, a cross-process
IPC read runs at 2108 GB/s against 2123 GB/s for a plain same-process copy, versus 369 GB/s for a
cross-device NCCL broadcast -- the bytes are not copied between processes at all, only mapped.

Two things follow from the fact that the peers share a device, and both make this smaller than the
NCCL engine rather than larger:

* No rendezvous exchange. The endpoint is derived from the GPU's UUID, which both peers can read on
  their own, so :meth:`build_topology` needs no metadata and the sampler's ``prepare`` result -- which
  the manager discards anyway -- is not needed.
* No shared rank space. Each pair gets a private channel, so a sending trainer rank is rank 0 of its
  own two-member channel rather than of one group spanning the job. With one sampler process per GPU
  that means every trainer rank sends, to the sampler on its own device, with no change to the caller
  and none of the rank-0 bottleneck the broadcast topology has -- each rank already materialises the
  full HF tensors. A tensor-parallel sampler is one process across several GPUs and so listens on only
  one of them; then one rank per sampler sends and the rest sit out. See :meth:`build_topology`.

ZMQ carries only handles and per-tensor metadata -- a few hundred bytes per bucket.
"""
import os
import pickle
import torch
import zmq
from typing import Any, AsyncGenerator, Generator

from twinkle import Platform, get_logger
from twinkle.utils.framework import Torch
from .base import CheckpointEngine

logger = get_logger()

# A trainer/sampler pair is a channel of exactly two: the trainer stages, the sampler maps.
_SENDER_RANK = 0
_RECEIVER_RANK = 1
# A trainer rank with no peer on its GPU, because a tensor-parallel sampler process covers several GPUs
# but listens on only one of them. It sends nothing; see build_topology.
_IDLE_RANK = -1
_CHANNEL_SIZE = 2
# Every tensor starts at a multiple of this. ``Tensor.view(dtype)`` requires the storage offset to be
# a multiple of the target element size, and 16 covers every dtype Torch has.
_ALIGNMENT = 16


class IPCCheckpointEngine(CheckpointEngine):
    """Hand weights to a sampler on the same device by mapping memory instead of copying it."""

    def __init__(self, bucket_size: int = 512 << 20, **kwargs) -> None:
        # Smaller default than the NCCL engine's 3 GB: a bigger bucket buys nothing when the transfer
        # is a mapping, and under colocation this buffer competes with the sampler for the same GPU.
        self.bucket_size = bucket_size

        # Set by the manager before prepare(); unused here, the endpoint is not negotiated.
        self.is_master = False

        self.rank: int | None = None
        self.world_size: int | None = None
        self.send_buf: torch.Tensor | None = None
        self.socket = None
        self._context = None
        self._handle = None
        self._shm = None
        # Receiver side: the mapping of the sender's buffer, kept across buckets. Re-mapping per
        # bucket is what makes device memory appear to grow during a sync.
        self._mapped: torch.Tensor | None = None
        self._mapped_shms = []
        self._mapped_signature = None

    # ── rendezvous ───────────────────────────────────────────────────────

    @staticmethod
    def endpoint() -> str:
        """The socket both peers derive independently from the device they share.

        The platform helper obtains the physical device UUID for the current local device.
        """
        uuid = str(Platform.get_vllm_device_uuid(Torch.get_current_device()))
        return f'ipc:///tmp/twinkle-colocate-{uuid}.sock'

    def prepare(self) -> dict[str, Any]:
        """Nothing to negotiate; the endpoint is derived, not exchanged."""
        return {}

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        """Decide which trainer ranks have a sampler to talk to, and pair them up.

        With one sampler process per GPU the two world sizes match, every trainer rank is rank 0 of its
        own channel, and all of them send -- there is no rank-0 bottleneck, since each rank already
        materialises the full HF tensors.

        A tensor-parallel sampler is a single process holding ``tp`` GPUs, so ``rollout_world_size``
        counts data-parallel replicas rather than devices and comes out ``tp`` times smaller. That
        process receives once and fans the weights out to its own tp workers itself, so only one trainer
        rank per sampler needs to send and the rest stay idle -- the same shape the NCCL engine uses,
        and no weights are lost, because every trainer rank holds the same full tensors anyway.

        Which rank that is follows from where the sampler listens. Measured: a process given
        ``CUDA_VISIBLE_DEVICES=0,1`` reports ``current_device()`` on the *first* of them, so its endpoint
        is that GPU's, and the trainer rank sharing that GPU is the one whose index is a multiple of
        ``tp``. Nothing here has to match them up explicitly: the endpoint is derived from the GPU, so
        the pairing falls out of both peers being on it.
        """
        if rollout_world_size <= 0 or trainer_world_size % rollout_world_size:
            raise ValueError(
                f'Colocated weight sync expects each sampler process to cover a whole number of trainer '
                f'ranks\' GPUs, got trainer={trainer_world_size} and rollout={rollout_world_size}. Unless '
                f'the two roles are placed on the same devices, this needs the NCCL engine instead.')
        # How many trainer ranks share one sampler process: 1 without tensor parallelism, tp with it.
        ranks_per_sampler = trainer_world_size // rollout_world_size
        trainer_kwargs = {
            'rank': [
                _SENDER_RANK if rank % ranks_per_sampler == 0 else _IDLE_RANK
                for rank in range(trainer_world_size)
            ],
            'world_size': [_CHANNEL_SIZE] * trainer_world_size,
            'master_metadata': [None] * trainer_world_size,
        }
        rollout_kwargs = {
            'rank': [_RECEIVER_RANK] * rollout_world_size,
            'world_size': [_CHANNEL_SIZE] * rollout_world_size,
            'master_metadata': [None] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int = _CHANNEL_SIZE, master_metadata=None):
        """Open the channel. PAIR, so a late peer queues rather than misses messages."""
        self.rank = rank
        self.world_size = world_size
        if rank == _IDLE_RANK:
            # No peer on this GPU, so no socket to open. Binding one would leave a path nobody reads.
            return
        endpoint = self.endpoint()
        self._context = zmq.Context()
        self.socket = self._context.socket(zmq.PAIR)
        if rank == _SENDER_RANK:
            # A socket file left by a crashed run would make bind fail; the sender owns the path.
            path = endpoint.removeprefix('ipc://')
            if os.path.exists(path):
                os.unlink(path)
            self.socket.bind(endpoint)
        else:
            self.socket.connect(endpoint)
        logger.info(f'IPC checkpoint engine rank {rank} on {endpoint}')

    def finalize(self):
        """Give the buffer back -- under colocation the sampler is waiting for this memory."""
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        if self.rank == _SENDER_RANK:
            path = self.endpoint().removeprefix('ipc://')
            if os.path.exists(path):
                os.unlink(path)
        if self._shm is not None:
            self.send_buf = None
            self._shm.close()
            self._shm.unlink()
            self._shm = None
        self.send_buf = None
        self._handle = None
        self._mapped = None
        for shm in self._mapped_shms:
            shm.close()
        self._mapped_shms.clear()
        self._mapped_signature = None
        self.rank = None

    # ── sender ───────────────────────────────────────────────────────────

    def _ensure_buffer(self, min_size: int) -> None:
        """Allocate the staging buffer, growing it if one tensor alone will not fit."""
        if self.send_buf is not None and self.send_buf.numel() >= min_size:
            return
        size = max(self.bucket_size, min_size)
        platform = Platform.get_platform()
        if platform.device_prefix() == 'npu' and not platform.is_ipc_supported():
            from multiprocessing import shared_memory

            if self._shm is not None:
                self.send_buf = None
                self._shm.close()
                self._shm.unlink()
                self._shm = None
            self._shm = shared_memory.SharedMemory(create=True, size=size)
            self.send_buf = torch.frombuffer(self._shm.buf, dtype=torch.uint8, count=size)
            self._handle = {'name': self._shm.name, 'size': size}
            return

        self.send_buf = torch.empty(
            size,
            dtype=torch.uint8,
            device=f'{platform.device_prefix()}:{Torch.get_current_device()}',
        )
        # One handle per buffer, reused for every bucket: the buffer is refilled, not reallocated, so
        # the mapping stays valid and the receiver can keep it.
        if platform.device_prefix() == 'npu':
            import torch_npu  # noqa: F401
        from torch.multiprocessing.reductions import reduce_tensor
        self._handle = reduce_tensor(self.send_buf)

    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Pack weights into the buffer bucket by bucket and tell the receiver where they are."""
        if self.rank == _IDLE_RANK:
            # Drained rather than ignored: the generator is what drives the bridge's export, and
            # abandoning it half-way would leave that work suspended mid-conversion.
            for _ in weights:
                pass
            return
        assert self.rank == _SENDER_RANK, 'Only the sender rank of a channel sends weights.'
        bucket_meta: list[dict] = []
        offset = 0
        total = 0

        for name, tensor in weights:
            flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            nbytes = flat.numel()
            start = (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            if bucket_meta and start + nbytes > (self.send_buf.numel() if self.send_buf is not None else 0):
                self._flush(bucket_meta, is_last=False)
                bucket_meta, offset, start = [], 0, 0
            self._ensure_buffer(start + nbytes)
            self.send_buf[start:start + nbytes].copy_(flat, non_blocking=True)
            bucket_meta.append({
                'name': name,
                'shape': tuple(tensor.shape),
                'dtype': tensor.dtype,
                'offset': start,
                'nbytes': nbytes,
            })
            offset = start + nbytes
            total += 1

        # Always a final message, even with nothing in it: it is what ends the receiver's loop.
        self._flush(bucket_meta, is_last=True)
        logger.info(f'IPC checkpoint engine sent {total} tensors')

    def _flush(self, bucket_meta: list[dict], is_last: bool) -> None:
        """Publish the filled part of the buffer and wait until the receiver is done with it."""
        # The copies above are non_blocking; without this the receiver could map bytes not yet written.
        Torch.synchronize()
        self.socket.send(pickle.dumps({'handle': self._handle, 'bucket_meta': bucket_meta, 'is_last': is_last}))
        # The receiver copies out of this buffer, so it must say so before we overwrite it.
        self.socket.recv()

    # ── receiver ─────────────────────────────────────────────────────────

    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Yield views onto the sender's buffer. Callers that keep a tensor must clone it."""
        assert self.rank is not None and self.rank != _SENDER_RANK, 'The sender rank does not receive.'
        while True:
            message = pickle.loads(self.socket.recv())
            bucket_meta = message['bucket_meta']
            # The closing message can be empty -- there is no buffer to map behind it, and for a model
            # with nothing to send it is the only message there is.
            if bucket_meta:
                buffer = self._map(message['handle'])
            for meta in bucket_meta:
                start, nbytes = meta['offset'], meta['nbytes']
                yield meta['name'], buffer[start:start + nbytes].view(meta['dtype']).view(meta['shape'])
            # Consumers copy with non_blocking=True, so the acknowledgement has to wait for the copies
            # and not merely for the loop above.
            Torch.synchronize()
            self.socket.send(b'ack')
            if message['is_last']:
                break

    def _map(self, handle) -> torch.Tensor:
        """Map the sender's buffer, reusing the mapping while the handle keeps describing it.

        Re-mapping on every bucket is what makes device memory look like it is growing during a sync,
        so the signature comparison here is load-bearing rather than an optimisation.
        """
        signature = self._handle_signature(handle)
        if self._mapped is not None and signature == self._mapped_signature:
            return self._mapped
        if isinstance(handle, dict):
            from multiprocessing import shared_memory

            mapped_shm = shared_memory.SharedMemory(name=handle['name'])
            self._mapped_shms.append(mapped_shm)
            self._mapped = torch.frombuffer(
                mapped_shm.buf,
                dtype=torch.uint8,
                count=handle['size'],
            )
            self._mapped_signature = signature
            return self._mapped

        func, args = handle
        args = list(args)
        if Platform.device_prefix() == 'npu':
            import torch_npu  # noqa: F401
        args[6] = Torch.get_current_device()
        self._mapped = func(*args)
        self._mapped_signature = signature
        return self._mapped

    @staticmethod
    def _handle_signature(handle) -> tuple:
        """Identify the memory a handle refers to, by the parts of it that compare cleanly.

        Locally implemented rather than shared with the sampler's worker extension, which has the same
        helper: the sampler imports this package, so importing it back would be circular.
        """
        if isinstance(handle, dict):
            return tuple(handle.items())
        _, args = handle
        return tuple((type(v).__name__, bytes(v) if isinstance(v, (bytes, bytearray)) else v) for v in args
                     if isinstance(v, (bytes, bytearray, int, float, bool, str)) or v is None)
