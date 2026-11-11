# Copyright (c) ModelScope Contributors. All rights reserved.
"""One bucketed IPC sender for local samplers and colocated training workers."""

import asyncio
import contextlib
import os
import re
import uuid
from typing import Optional

import torch

from twinkle import Platform, get_logger
from twinkle.checkpoint_engine.zmq_auth import send_auth_pyobj
from twinkle.utils.framework import Torch
from twinkle.utils.zmq_utils import configure_zmq_socket, get_timeout_s_from_env

logger = get_logger()


class VLLMWeightSender:
    """Keep a reusable device bucket in the process that owns the source weights.

    With ``engine`` set, launch the receiver through the local AsyncLLM. A remote
    colocated sender instead receives an endpoint and the manager starts the
    vLLM receivers separately; the sampler actor carries control messages only.
    """

    def __init__(self, engine=None):
        self.engine = engine
        self._ipc_buffer = None
        self._ipc_handle = None
        self._ipc_buffer_size = 0

    async def _send_empty(self, endpoint, auth_key):
        """Release an already-submitted external receiver for an empty export."""
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        timeout = get_timeout_s_from_env("TWINKLE_VLLM_IPC_TIMEOUT_S", 300)
        configure_zmq_socket(socket, timeout_ms=timeout * 1000, linger=0)
        socket.bind(endpoint)
        try:

            def handshake():
                send_auth_pyobj(socket, {"empty": True}, auth_key)
                socket.recv()

            await asyncio.get_running_loop().run_in_executor(None, handshake)
        finally:
            socket.close()
            context.term()
            if endpoint.startswith("ipc://") and os.path.exists(endpoint[6:]):
                os.remove(endpoint[6:])

    async def update_weights(
        self,
        weights,
        peft_config: dict | None = None,
        base_sync_done: bool = False,
        bucket_size_mb: int = 2048,
        zmq_handle: str | None = None,
        auth_key: str = "",
        device: torch.device | None = None,
        direct_large_tensors: bool = False,
        **kwargs,
    ) -> None:
        """Update model weights via ZMQ + device IPC to worker extension.

        Accepts **either** a ``dict[str, Tensor]`` (legacy) **or** an async
        generator / sync generator of ``(name, tensor)`` pairs (streaming).

        The streaming path avoids accumulating a full model copy on GPU:
        tensors are consumed one-by-one from the generator, copied into a
        device IPC bucket, and flushed to the vLLM worker subprocess when the
        bucket is full.

        Args:
            weights: Weights to transfer.  ``dict[str, Tensor]`` or
                ``(Async)Generator[tuple[str, Tensor], ...]``.
            peft_config: PEFT config dict for LoRA adapter loading.
            base_sync_done: If True with peft_config, load as LoRA adapter.
            bucket_size_mb: Size of transfer bucket in MB.
        """
        import asyncio
        import gc
        import time

        import zmq

        start_time = time.time()

        # Normalise *weights* into an async iterator regardless of input type.
        if isinstance(weights, dict):

            async def _dict_iter():
                for item in weights.items():
                    yield item

            weight_aiter = _dict_iter()
        elif hasattr(weights, "__aiter__"):
            weight_aiter = weights.__aiter__()
        else:
            # sync generator / iterable
            async def _sync_iter():
                for item in weights:
                    yield item

            weight_aiter = _sync_iter()

        # Peek first tensor to detect device (supported accelerator → IPC, CPU → SHM).
        try:
            first_name, first_tensor = await weight_aiter.__anext__()
        except StopAsyncIteration:
            logger.warning("update_weights called with empty weights")
            if self.engine is None and zmq_handle is not None:
                await self._send_empty(zmq_handle, auth_key)
            return

        # Initial HF export can still be on the host, before lazy FSDP wrapping.
        # A direct receiver always maps a device bucket; stage host tensors into
        # that bucket rather than silently switching its wire format to SHM.
        transfer_device = first_tensor.device if device is None else torch.device(device)
        use_device_ipc = transfer_device.type == "cuda"
        if transfer_device.type == "npu":
            use_device_ipc = Platform.get_platform().supports_device_ipc()
        use_shm = not use_device_ipc

        # Use a per-sync unique IPC endpoint to avoid cross-actor collisions
        # when multiple sampler actors share the same device UUID.
        device_uuid = Platform.get_vllm_device_uuid(0)
        sync_id = uuid.uuid4().hex
        zmq_handle = zmq_handle or f"ipc:///tmp/twinkle-ipc-{device_uuid}-{os.getpid()}-{sync_id}.sock"

        env_bucket_mb = os.environ.get("TWINKLE_VLLM_BUCKET_SIZE_MB")
        if env_bucket_mb is not None:
            bucket_size_mb = int(env_bucket_mb)
        if bucket_size_mb <= 0:
            raise ValueError(f"bucket_size_mb must be > 0, got {bucket_size_mb}")

        bucket_size = bucket_size_mb << 20
        lora_mode = bool(base_sync_done and peft_config)

        # Create transfer buffer
        buffer = None
        shm = None

        if use_device_ipc:
            # Register the platform's tensor reducer (no-op where the CUDA one is builtin).
            Platform.get_platform().ensure_ipc_reducer()
            from torch.multiprocessing.reductions import reduce_tensor

            # Reuse a long-lived IPC bucket whenever the requested size
            # fits. The handle is produced once and shipped to every
            # subsequent sync so each worker's ``shared_cache`` stays warm
            # and no new device IPC mapping is created per sync.
            need_realloc = (
                self._ipc_buffer is None
                or self._ipc_buffer_size < bucket_size
                or self._ipc_buffer.device != transfer_device
            )
            if need_realloc:
                # Drop the old handle/buffer before allocating a bigger one
                # so we do not briefly hold both and double the peak usage.
                self._ipc_buffer = None
                self._ipc_handle = None
                self._ipc_buffer_size = 0
                self._ipc_buffer = torch.empty(bucket_size, dtype=torch.uint8, device=transfer_device)
                self._ipc_handle = reduce_tensor(self._ipc_buffer)
                self._ipc_buffer_size = bucket_size
            buffer = self._ipc_buffer
            ipc_handle = self._ipc_handle
        else:
            from multiprocessing import shared_memory

            shm_name = f"twinkle_weights_{uuid.uuid4().hex}"
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=bucket_size)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

        # Setup ZMQ socket FIRST (bind before worker connects)
        zmq_ctx = zmq.Context()
        socket = zmq_ctx.socket(zmq.REQ)
        zmq_timeout_s = get_timeout_s_from_env("TWINKLE_VLLM_IPC_TIMEOUT_S", 300)
        configure_zmq_socket(socket, timeout_ms=zmq_timeout_s * 1000, linger=0)
        socket.bind(zmq_handle)

        loop = asyncio.get_running_loop()

        # Non-blocking ZMQ helpers — run blocking socket ops in the
        # default executor so the event loop stays responsive.  This is
        # critical when TP > 1: collective_rpc is an async task on the
        # same loop, and blocking socket.recv() would prevent it from
        # being scheduled, causing a deadlock.
        def _zmq_send_recv(payload, where: str):
            try:
                send_auth_pyobj(socket, payload, auth_key)
                return socket.recv()
            except zmq.error.Again as e:
                raise RuntimeError(f"IPC timeout ({zmq_timeout_s}s) during {where} on {zmq_handle}") from e

        n_weights = 0
        worker_task: asyncio.Future | None = None
        try:
            # A direct trainer sender has no AsyncLLM in this process: its
            # manager has already submitted the remote receiver RPC.
            if self.engine is not None:
                worker_task = asyncio.ensure_future(
                    self.engine.collective_rpc(
                        "update_weights_from_ipc",
                        kwargs={
                            "peft_config": peft_config,
                            "base_sync_done": base_sync_done,
                            "use_shm": use_shm,
                            "zmq_handle": zmq_handle,
                            "auth_key": auth_key,
                        },
                    )
                )

            # Send IPC/SHM handle, wait for worker ready (non-blocking)
            handle_payload = ipc_handle if use_device_ipc else {"name": shm_name, "size": bucket_size}
            await loop.run_in_executor(None, _zmq_send_recv, handle_payload, "handle handshake")

            # Stream weights into buckets and send to worker
            async def _chain_first():
                """Re-inject the peeked first tensor, then yield the rest."""
                yield first_name, first_tensor
                async for item in weight_aiter:
                    yield item

            offset = 0
            bucket_meta: list[dict] = []
            current_expert_layer: str | None = None

            def _get_expert_layer_prefix(weight_name: str) -> str | None:
                m = re.match(r"^(.*\.mlp\.experts)\.\d+\.", weight_name)
                return m.group(1) if m else None

            async def _flush_bucket(is_last: bool) -> None:
                nonlocal offset, bucket_meta, current_expert_layer
                if not bucket_meta and not is_last:
                    return
                if buffer.device.type != "cpu":
                    Torch.synchronize()
                await loop.run_in_executor(
                    None,
                    _zmq_send_recv,
                    {
                        "bucket_meta": bucket_meta,
                        "is_last": is_last,
                    },
                    "final bucket" if is_last else "bucket flush",
                )
                offset = 0
                bucket_meta = []
                current_expert_layer = None

            async for name, weight in _chain_first():
                if use_shm and weight.device.type != "cpu":
                    weight = weight.cpu()
                if not weight.is_contiguous():
                    weight = weight.contiguous()

                weight_u8 = weight.view(-1).view(torch.uint8)
                total_nbytes = int(weight_u8.numel())
                if (direct_large_tensors and use_device_ipc and not lora_mode
                        and weight.device == transfer_device and total_nbytes > bucket_size):
                    await _flush_bucket(is_last=False)
                    Torch.synchronize(transfer_device)
                    await loop.run_in_executor(None, _zmq_send_recv, {
                        'bucket_meta': [{'name': name, 'shape': weight.shape, 'dtype': weight.dtype,
                                         'handle': reduce_tensor(weight)}], 'is_last': False,
                    }, 'large tensor IPC')
                    n_weights += 1
                    continue
                expert_layer_prefix = _get_expert_layer_prefix(name) if lora_mode else None
                if lora_mode and offset > 0:
                    # Keep each expert layer in an isolated bucket to avoid sending
                    # partial expert-layer weights.
                    if current_expert_layer != expert_layer_prefix:
                        await _flush_bucket(is_last=False)
                if lora_mode:
                    current_expert_layer = expert_layer_prefix

                chunk_offset = 0
                while chunk_offset < total_nbytes:
                    if offset >= bucket_size:
                        await _flush_bucket(is_last=False)

                    chunk_nbytes = min(bucket_size - offset, total_nbytes - chunk_offset)
                    buffer[offset : offset + chunk_nbytes].copy_(
                        weight_u8[chunk_offset : chunk_offset + chunk_nbytes],
                        non_blocking=True,
                    )
                    bucket_meta.append(
                        {
                            "name": name,
                            "shape": weight.shape,
                            "dtype": weight.dtype,
                            "offset": offset,
                            "nbytes": chunk_nbytes,
                            "chunk_offset": chunk_offset,
                            "total_nbytes": total_nbytes,
                        }
                    )
                    offset += chunk_nbytes
                    chunk_offset += chunk_nbytes
                n_weights += 1

            # Send last bucket
            await _flush_bucket(is_last=True)

            # Wait for worker to finish loading
            if worker_task is not None:
                await worker_task
        finally:
            # Clean up — always release resources regardless of exceptions
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
                with contextlib.suppress(BaseException):
                    await worker_task
            socket.close()
            zmq_ctx.term()
            if zmq_handle.startswith("ipc://"):
                ipc_path = zmq_handle[len("ipc://") :]
                try:
                    if os.path.exists(ipc_path):
                        os.remove(ipc_path)
                except OSError:
                    pass
            del buffer
            if shm is not None:
                shm.close()
                shm.unlink()
                del shm
            if not direct_large_tensors:
                gc.collect()

        elapsed = time.time() - start_time
        mode = "LoRA" if base_sync_done and peft_config else "base"
        logger.info(f"Updated {n_weights} {mode} weights via {'IPC' if use_device_ipc else 'SHM'} in {elapsed:.2f}s")
