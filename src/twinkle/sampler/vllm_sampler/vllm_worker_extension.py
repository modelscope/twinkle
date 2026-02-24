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
import ctypes
import gc
import os
import platform
import re
import signal
import torch
from typing import Dict, List, Optional, Tuple

from twinkle import get_logger
from twinkle.utils import Platform
from twinkle.utils.framework import Torch

logger = get_logger()


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != 'Linux':
        return
    libc = ctypes.CDLL('libc.so.6')
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


# Constants for the RL training LoRA adapter identity.
VLLM_LORA_INT_ID = 111
VLLM_LORA_NAME = 'twinkle_lora'
VLLM_LORA_PATH = 'twinkle_lora_path'


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


class TwinkleWorkerExtension:
    """Extension class for vLLM workers to support weight synchronization.

    Mixed into vLLM's Worker class via ``worker_extension_cls``.  Methods
    are called from the vLLMSampler Ray actor through
    ``AsyncLLM.collective_rpc()``.

    Usage:
        worker_extension_cls="twinkle.sampler.vllm_sampler.vllm_worker_extension.TwinkleWorkerExtension"
    """

    def __new__(cls, *args, **kwargs):
        from twinkle.patch.vllm_lora_weights import VLLMLoraWeights
        set_death_signal()
        VLLMLoraWeights()(None)

        return super().__new__(cls)

    def monkey_patch_model(self):
        from twinkle.patch.vllm_moe_loader import VLLMMoEWeights
        VLLMMoEWeights()(self.model_runner.model)

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

        For TP > 1, only TP rank 0 communicates with the VLLMEngine over
        ZMQ.  It broadcasts the IPC handle and bucket metadata to other
        ranks via ``torch.distributed``, so every rank can read the shared
        buffer and call ``load_weights`` for its own TP shard.

        Args:
            peft_config: If provided with base_sync_done, loads as LoRA.
            base_sync_done: If True and peft_config, replaces existing LoRA.
            use_shm: If True, use shared memory instead of CUDA IPC.
        """
        import torch.distributed as dist
        import zmq

        if self.device is None:
            # fix: In some worker paths, omitting local_rank can pick the wrong device / trigger get_device arg issues.
            # fix: Pass local_rank when available so each worker binds to the expected local device.
            print(f"VLLM Worker local_rank: {getattr(self, 'local_rank', None)} <<<<<<<<<<<<< {Torch.get_device()}")
            self.device = torch.device(Torch.get_device(getattr(self, 'local_rank', None)))

        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        # Detect TP rank — vLLM sets self.rank on each worker.
        tp_rank = getattr(self, 'rank', 0)
        tp_size = 1
        try:
            tp_size = self.model_runner.parallel_config.tensor_parallel_size
        except Exception:
            pass

        is_driver = (tp_rank == 0)

        if tp_size > 1:
            # Use vLLM's built-in TP cpu group for object broadcasts.
            from vllm.distributed import get_tp_group
            tp_coord = get_tp_group()
            cpu_group = tp_coord.cpu_group
            broadcast_src = tp_coord.ranks[0]  # global rank of TP rank 0
        else:
            cpu_group = None
            broadcast_src = 0

        def _broadcast_obj(obj):
            """Broadcast a picklable object from TP rank 0 to all TP ranks."""
            obj_list = [obj]
            dist.broadcast_object_list(obj_list, src=broadcast_src, group=cpu_group)
            return obj_list[0]

        # ── Step 1: Establish ZMQ connection (driver only) ──
        socket = None
        if is_driver:
            if not hasattr(self, '_zmq_ctx') or self._zmq_ctx is None:
                self._zmq_ctx = zmq.Context()
            socket = self._zmq_ctx.socket(zmq.REP)
            socket.connect(self._get_zmq_handle())

        # ── Step 2: Receive and broadcast IPC/SHM handle ──
        buffer, shm = None, None

        if is_driver:
            comm_metadata = socket.recv_pyobj()
        else:
            comm_metadata = None

        if tp_size > 1:
            comm_metadata = _broadcast_obj(comm_metadata)

        if not use_shm:
            handle = comm_metadata
            # All TP ranks rebuild the IPC buffer from the same handle.
            # CUDA IPC allows any process on the same node to map the memory.
            buffer = _rebuild_ipc(handle, self.device.index)
        else:
            from multiprocessing import shared_memory
            buffer, shm = _rebuild_shared_memory(
                comm_metadata['name'],
                comm_metadata['size'],
            )

        if is_driver:
            socket.send(b'')  # Ready

        # ── Step 3: Receive and process weight buckets ──
        while True:
            # Only the driver receives bucket metadata from VLLMEngine.
            if is_driver:
                metadata = socket.recv_pyobj()
            else:
                metadata = None

            if tp_size > 1:
                metadata = _broadcast_obj(metadata)

            weights = []
            for name, meta in metadata['bucket_meta'].items():
                shape, dtype, offset = meta['shape'], meta['dtype'], meta['offset']
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset:offset + size].view(dtype=dtype).view(shape)
                if not use_shm:
                    tensor = tensor.clone()
                else:
                    tensor = tensor.to(self.device)
                weights.append((name, tensor))

            Torch.synchronize()

            if is_driver:
                socket.send(b'')

            # Ensure all ranks finish reading the buffer before the driver
            # proceeds to the next bucket (which overwrites the buffer).
            if tp_size > 1:
                dist.barrier(group=cpu_group)

            self._load_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
            del weights

            if metadata['is_last']:
                break

        if is_driver and socket is not None:
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
            # fix: Keep device resolution consistent with update_weights_from_ipc to avoid path divergence.
            self.device = torch.device(Torch.get_device(getattr(self, 'local_rank', None)))

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

        PEFT names look like:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        vLLM expects:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

        Only the adapter-name segment (e.g. ``.default.``) between
        ``lora_A``/``lora_B`` and ``weight`` needs to be removed.
        """
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

            converted = {self._convert_peft_to_vllm_lora_name(n): t for n, t in weights}
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
                getattr(self, 'vllm_config', None),
                'lora_config',
                None,
            ) is not None

            # When vLLM LoRA is enabled, some LinearBase modules are
            # replaced by *WithLoRA wrappers.  Their parameters shift
            # from e.g. ``gate.weight`` to ``gate.base_layer.weight``.
            # HF checkpoint names do NOT contain ``.base_layer.``, so
            # vLLM's own ``load_weights`` will KeyError on them.
            #
            # Build a set of base-layer prefixes that need rewriting.
            lora_base_prefixes: set = set()
            if vllm_has_lora:
                from vllm.lora.layers import BaseLayerWithLoRA
                for mod_name, mod in self.model_runner.model.named_modules():
                    if isinstance(mod, BaseLayerWithLoRA):
                        # mod_name is e.g. "model.layers.0.mlp.gate"
                        lora_base_prefixes.add(mod_name + '.')

            converted = []
            for name, tensor in weights:
                if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                    continue
                name = name.removeprefix('model.base_model.model.')
                name = name.removeprefix('base_model.model.')
                if not vllm_has_lora:
                    name = name.replace('.base_layer.', '.')
                else:
                    # Insert ``.base_layer.`` for weights whose module
                    # has been wrapped by LoRA and whose name does NOT
                    # already contain it.
                    if '.base_layer.' not in name:
                        for pfx in lora_base_prefixes:
                            if name.startswith(pfx):
                                # e.g. "model.layers.0.mlp.gate.weight"
                                # →    "model.layers.0.mlp.gate.base_layer.weight"
                                suffix = name[len(pfx):]
                                name = pfx + 'base_layer.' + suffix
                                break
                converted.append((name, tensor))

            if not converted:
                return

            self.model_runner.model.load_weights(converted)
            logger.info(f'Loaded {len(converted)} base weights')

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for IPC communication."""
        if not hasattr(self, '_device_uuid') or not self._device_uuid:
            # fix: Always use platform fallback to avoid worker-side crashes when NPU get_device_uuid is unimplemented.
            self._device_uuid = Platform.get_vllm_device_uuid(self.device.index)
        return f'ipc:///tmp/twinkle-ipc-{self._device_uuid}.sock'
