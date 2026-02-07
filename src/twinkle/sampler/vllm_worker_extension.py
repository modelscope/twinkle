# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM Worker Extension for weight synchronization.

This module provides a Worker extension class that enables weight
synchronization from training to vLLM inference workers via collective_rpc.

The extension class is injected into vLLM workers via the `worker_extension_cls`
parameter and provides methods for:
- Direct weight loading via model.load_weights()
- LoRA adapter loading via add_lora()

Reference: verl's vLLMColocateWorkerExtension implementation.
"""
import gc
import logging
import os
import platform
import ctypes
import re
import signal
from typing import Dict, List, Optional, Tuple

import torch
from twinkle.utils.framework import Torch

logger = logging.getLogger(__name__)


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


# Constants for the RL training LoRA adapter identity.
VLLM_LORA_INT_ID = 1
VLLM_LORA_NAME = "twinkle_lora"
VLLM_LORA_PATH = "twinkle_lora_path"


def _rebuild_ipc(handle, device_id: Optional[int] = None) -> torch.Tensor:
    """Rebuild CUDA tensor from IPC handle."""
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    func, args = handle
    list_args = list(args)
    if device_id is not None:
        list_args[6] = device_id

    if callable(func):
        return func(*list_args)
    else:
        return rebuild_cuda_tensor(*list_args)


def _rebuild_shared_memory(name: str, size: int):
    """Rebuild tensor from shared memory.  Returns (tensor, shm)."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
    return tensor, shm


def _get_device_uuid(device_id: int) -> str:
    """Get unique device identifier."""
    from vllm.platforms import current_platform
    return current_platform.get_device_uuid(device_id)


class TwinkleWorkerExtension:
    """Extension class for vLLM workers to support weight synchronization.

    Mixed into vLLM's Worker class via ``worker_extension_cls``.  Methods
    are called from the VLLMSampler Ray actor through
    ``AsyncLLM.collective_rpc()``.

    Usage:
        worker_extension_cls="twinkle.sampler.vllm_worker_extension.TwinkleWorkerExtension"
    """

    def __new__(cls, *args, **kwargs):
        from twinkle.patch.vllm_lora_weights import VLLMLoraWeights
        set_death_signal()
        VLLMLoraWeights()(None)
        return super().__new__(cls)

    # -----------------------------------------------------------------
    # Public API — called via collective_rpc from VLLMEngine
    # -----------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        peft_config: Optional[Dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
    ) -> None:
        """Receive and load weights via ZMQ + CUDA IPC/SHM.

        Called via ``collective_rpc("update_weights_from_ipc", ...)`` from
        :meth:`VLLMEngine.update_weights`.  The VLLMEngine sends weights
        in buckets over a ZMQ REQ/REP channel backed by CUDA IPC (GPU
        tensors) or shared memory (CPU tensors).

        Args:
            peft_config: If provided with base_sync_done, loads as LoRA.
            base_sync_done: If True and peft_config, replaces existing LoRA.
            use_shm: If True, use shared memory instead of CUDA IPC.
        """
        import zmq

        if self.device is None:
            self.device = torch.device(Torch.get_device())

        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        # Setup ZMQ socket
        if not hasattr(self, '_zmq_ctx') or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(self._get_zmq_handle())

        comm_metadata = socket.recv_pyobj()

        buffer, shm = None, None
        if not use_shm:
            handle = comm_metadata
            buffer = _rebuild_ipc(handle, self.device.index)
        else:
            from multiprocessing import shared_memory
            buffer, shm = _rebuild_shared_memory(
                comm_metadata["name"], comm_metadata["size"],
            )

        socket.send(b"")  # Ready

        while True:
            metadata = socket.recv_pyobj()
            weights = []

            for name, meta in metadata["bucket_meta"].items():
                shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset:offset + size].view(dtype=dtype).view(shape)
                if not use_shm:
                    tensor = tensor.clone()
                else:
                    tensor = tensor.to(self.device)
                weights.append((name, tensor))

            Torch.synchronize()
            socket.send(b"")

            self._load_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
            del weights

            if metadata["is_last"]:
                break

        socket.close()
        del buffer
        if shm is not None:
            shm.close()
            del shm
        gc.collect()
        Torch.ipc_collect()
        Torch.empty_cache()

    def load_synced_weights(
        self,
        weights: Dict[str, torch.Tensor],
        peft_config: Optional[Dict] = None,
        base_sync_done: bool = False,
    ) -> None:
        """Load weights received from the checkpoint engine.

        Called via ``collective_rpc("load_synced_weights", kwargs=...)``
        from :meth:`VLLMEngine.update_weights`.

        Two modes:
        - **Base model** (``base_sync_done=False``):
          Strips PEFT prefixes and loads via ``model.load_weights()``.
        - **LoRA adapter** (``base_sync_done=True`` + ``peft_config``):
          Converts names to vLLM LoRA format and loads via ``add_lora()``.

        Args:
            weights: Dict mapping weight names to tensors.
            peft_config: PEFT config dict for LoRA adapter loading.
            base_sync_done: If True with peft_config, load as LoRA adapter.
        """
        if self.device is None:
            self.device = torch.device(Torch.get_device())

        weight_list = list(weights.items())
        self._load_weights(weight_list, peft_config=peft_config, base_sync_done=base_sync_done)

        gc.collect()
        Torch.empty_cache()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _convert_peft_to_vllm_lora_name(name: str) -> str:
        """Convert PEFT LoRA weight name to vLLM format.

        PEFT: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        vLLM: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        """
        if name.startswith('base_model.model.model.'):
            name = 'base_model.model.' + name[len('base_model.model.model.'):]
        name = re.sub(r'\.lora_A\.[^.]+\.', '.lora_A.', name)
        name = re.sub(r'\.lora_B\.[^.]+\.', '.lora_B.', name)
        return name

    # Stacked parameter mapping matching vLLM Qwen2 model:
    # (stacked_param_name, source_shard_name, shard_id)
    def _load_weights(
        self,
        weights: List[Tuple[str, torch.Tensor]],
        peft_config: Optional[Dict],
        base_sync_done: bool,
    ) -> None:
        """Load a batch of weights into vLLM.

        Two modes:
        - LoRA mode (``peft_config`` and ``base_sync_done``): Loads weights as
          a tensor-based LoRA adapter via ``add_lora()``.
        - Base model mode: Strips PEFT prefixes, merges split weights
          (q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj) into vLLM's
          stacked format, normalizes prefixes, then loads via direct param copy.
        """
        if peft_config and base_sync_done:
            # Remove existing LoRA before replacing
            self.remove_lora(VLLM_LORA_INT_ID)

            from twinkle.patch.vllm_lora_weights import TensorLoRARequest

            converted = {
                self._convert_peft_to_vllm_lora_name(n): t
                for n, t in weights
            }
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=converted,
            )
            self.add_lora(lora_request)
        else:
            # Base model mode — strip PEFT prefixes and delegate to
            # vLLM's model.load_weights() which handles stacked params,
            # prefix normalization, and weight_loader internally.
            vllm_has_lora = getattr(
                getattr(self, 'vllm_config', None), 'lora_config', None,
            ) is not None

            converted = []
            for name, tensor in weights:
                if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                    continue
                name = name.removeprefix('model.base_model.model.')
                name = name.removeprefix('base_model.model.')
                if not vllm_has_lora:
                    name = name.replace('.base_layer.', '.')
                converted.append((name, tensor))

            if not converted:
                return

            self.model_runner.model.load_weights(converted)
            logger.info(f"Loaded {len(converted)} base weights")

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for IPC communication."""
        if not hasattr(self, '_device_uuid') or not self._device_uuid:
            self._device_uuid = _get_device_uuid(self.device.index)
        return f"ipc:///tmp/twinkle-ipc-{self._device_uuid}.sock"
