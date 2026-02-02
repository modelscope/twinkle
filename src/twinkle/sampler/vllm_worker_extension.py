# Copyright (c) ModelScope Contributors. All rights reserved.
"""vLLM Worker Extension for colocated training.

This module provides a Worker extension class that enables direct weight
synchronization from training to vLLM inference workers.

The extension class is injected into vLLM workers via the `worker_extension_cls`
parameter and provides methods for:
- Direct weight loading via model.load_weights()
- CUDA IPC weight transfer (colocate mode)
- LoRA adapter loading
- Weight synchronization coordination

Reference: verl's vLLMColocateWorkerExtension implementation.
"""
import gc
import hashlib
import logging
import os
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from twinkle.utils.framework import Framework, Torch

logger = logging.getLogger(__name__)

# TODO: get from tenant context
VLLM_LORA_INT_ID = 1
VLLM_LORA_NAME = "twinkle_lora"
VLLM_LORA_PATH = "twinkle_lora_path"

def rebuild_ipc(handle: Tuple[Callable, tuple], device_id: Optional[int] = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # Change device ID for different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    return func(*list_args)


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)
    return tensor, shm


def get_device_uuid(device_id: int) -> str:
    """Get unique device identifier."""
    from vllm.platforms import current_platform
    return current_platform.get_device_uuid(device_id)


class TwinkleWorkerExtension:
    """
    Extension class for vLLM workers to support weight synchronization.
    
    This class is designed to be mixed into vLLM's Worker class via the
    `worker_extension_cls` parameter. It provides direct access to the
    model's load_weights method for efficient weight synchronization.

    Usage:
        When creating VLLMEngine, pass:
        worker_extension_cls="twinkle.sampler.vllm_worker_extension.TwinkleWorkerExtension"
    """
    
    def update_weights_from_tensors(
        self,
        weights: List[Tuple[str, torch.Tensor]],
    ) -> int:
        # do we need searialization for tensors?
        # do we need bucket loading?
        model = self.model_runner.model
        
        try:
            # Call model's load_weights directly
            loaded_params = model.load_weights(weights)
            logger.info(f"Loaded {len(loaded_params)} weight tensors directly")
            return len(loaded_params)
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            raise

    def update_weights_from_ipc(
        self,
        peft_config: Optional[Dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
    ) -> None:
        """
        Update weights via CUDA IPC or shared memory.
        
        This method receives weights from training process via ZMQ + IPC.
        Only works in colocate mode (same machine).
        
        Args:
            peft_config: If provided, loads as LoRA adapter.
            base_sync_done: If True and peft_config provided, replaces existing LoRA.
            use_shm: If True, use shared memory instead of CUDA IPC.
        """
        import zmq
        
        # Get device info
        if self.device is None:
            # ascend vllm does not set device, set here
            self.device = torch.device(Torch.get_device())
        
        # remove existing LoRA if present
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        # Setup ZMQ socket
        if not hasattr(self, '_zmq_ctx') or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(self._get_zmq_handle())

        comm_metadata = socket.recv_pyobj()
        buffer, shm = None, None
        if not use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        
        socket.send(b"")  # Ready to receive
        
        # Receive and load weights in buckets
        while True:
            metadata = socket.recv_pyobj()
            weights = []
            
            for name, meta in metadata["bucket_meta"].items():
                shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset:offset + size].view(dtype=dtype).view(shape)
                
                if not use_shm:
                    # CUDA IPC: clone to release IPC memory
                    tensor = tensor.clone()
                else:
                    tensor = tensor.to(self.device)
                weights.append((name, tensor))
            
            Torch.synchronize()
            socket.send(b"")
            
            # Load weights
            self._update_weights(weights, peft_config=peft_config, base_sync_done=base_sync_done)
            del weights
            
            if metadata["is_last"]:
                break
        
        # Cleanup
        socket.close()
        del buffer
        if shm is not None:
            shm.close()
            del shm
        gc.collect()
        Torch.ipc_collect()
        Torch.empty_cache()
    
    def _update_weights(
        self,
        weights: List[Tuple[str, torch.Tensor]],
        peft_config: Optional[Dict],
        base_sync_done: bool,
    ) -> None:
        """Load a batch of weights."""
        if peft_config and base_sync_done:
            # LoRA mode
            from twinkle.patch.vllm_lora_weights import TensorLoRARequest
            
            weights_dict = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights_dict,
            )
            self.add_lora(lora_request)
        else:
            # TODO: FP8 support
            self.model_runner.model.load_weights(weights)
    
    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for IPC communication."""
        if not hasattr(self, 'device_uuid') or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/twinkle-ipc-{self.device_uuid}.sock"
