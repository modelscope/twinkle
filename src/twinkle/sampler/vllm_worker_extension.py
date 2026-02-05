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
    
    def _ensure_lora_patch_applied(self):
        """Ensure VLLMLoraWeights patch is applied for tensor-based LoRA loading."""
        if getattr(self, '_lora_patch_applied', False):
            return
        
        # Apply patch directly to LRUCacheWorkerLoRAManager._load_adapter
        # This is a simplified version that doesn't need sampler reference
        # The full VLLMLoraWeights.patch() needs sampler for tokenizer, but
        # for _load_adapter patch we only need the core tensor loading logic
        
        from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
        try:
            from vllm.lora.models import LoRAModel
        except ImportError:
            from vllm.lora.lora_model import LoRAModel
        from vllm.lora.utils import get_adapter_absolute_path
        from vllm.lora.peft_helper import PEFTHelper
        from twinkle.patch.vllm_lora_weights import TensorLoRARequest
        
        def patched_load_adapter(manager: LRUCacheWorkerLoRAManager, lora_request) -> LoRAModel:
            """Load LoRA adapter, supporting tensor-based loading for TensorLoRARequest."""
            supported_lora_modules = manager._adapter_manager.supported_lora_modules
            packed_modules_mapping = manager._adapter_manager.packed_modules_mapping
            expected_lora_modules: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)
            expected_lora_modules = list(set(expected_lora_modules))
            
            lora_tensors = None
            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(lora_path, manager.max_position_embeddings)
            
            peft_helper.validate_legal(manager.lora_config)
            model = manager._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, 'hf_to_vllm_mapper', None)
            
            if isinstance(lora_request, TensorLoRARequest):
                lora = manager._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device='cpu',
                    dtype=manager.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=manager.vocab_size + manager.lora_config.lora_extra_vocab_size,
                    embedding_modules=manager.embedding_modules,
                    embedding_padding_modules=manager.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            else:
                lora = manager._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device='cpu',
                    dtype=manager.lora_config.lora_dtype,
                    target_embedding_padding=manager.vocab_size + manager.lora_config.lora_extra_vocab_size,
                    embedding_modules=manager.embedding_modules,
                    embedding_padding_modules=manager.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            
            if lora.extra_vocab_size > manager.lora_config.lora_extra_vocab_size:
                raise ValueError(f'LoRA added vocab size {lora.extra_vocab_size} is greater than '
                                f'lora_extra_vocab_size {manager.lora_config.lora_extra_vocab_size}.')
            return lora
        
        if not hasattr(LRUCacheWorkerLoRAManager, '_old_load_adapter'):
            LRUCacheWorkerLoRAManager._old_load_adapter = LRUCacheWorkerLoRAManager._load_adapter
            LRUCacheWorkerLoRAManager._load_adapter = patched_load_adapter
        
        self._lora_patch_applied = True
        logger.info("LoRA tensor loading patch applied to LRUCacheWorkerLoRAManager")
    
    def _convert_peft_to_vllm_lora_name(self, name: str) -> str:
        """Convert PEFT LoRA weight name to vLLM format.
        
        PEFT format: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
        vLLM format: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
        
        Transformations:
        1. base_model.model.model.X -> base_model.model.X (remove extra model.)
        2. lora_A.default.weight -> lora_A.weight (remove .default)
        """
        # Remove extra 'model.' prefix if present
        if name.startswith('base_model.model.model.'):
            name = 'base_model.model.' + name[len('base_model.model.model.'):]
        
        # Remove '.default' from LoRA weight names
        # e.g., lora_A.default.weight -> lora_A.weight
        name = name.replace('.lora_A.default.', '.lora_A.')
        name = name.replace('.lora_B.default.', '.lora_B.')
        
        return name
    
    def _update_weights(
        self,
        weights: List[Tuple[str, torch.Tensor]],
        peft_config: Optional[Dict],
        base_sync_done: bool,
    ) -> None:
        """Load a batch of weights."""
        if peft_config and base_sync_done:
            # LoRA mode - need patch for tensor-based loading
            self._ensure_lora_patch_applied()
            
            from twinkle.patch.vllm_lora_weights import TensorLoRARequest
            
            # Convert PEFT weight names to vLLM format
            converted_weights = {}
            for name, tensor in weights:
                vllm_name = self._convert_peft_to_vllm_lora_name(name)
                converted_weights[vllm_name] = tensor
            
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=converted_weights,
            )
            self.add_lora(lora_request)
        else:
            # Strip PEFT prefix from weight names if present
            # PEFT uses 'base_model.model.model.' prefix while vLLM expects 'model.'
            # Also filter out LoRA-specific weights (lora_A, lora_B) as they should
            # be handled separately in LoRA mode
            converted_weights = []
            for name, tensor in weights:
                # Skip LoRA-specific weights for base model sync
                if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                    continue
                # Remove PEFT wrapper prefixes
                if name.startswith('base_model.model.model.'):
                    name = 'model.' + name[len('base_model.model.model.'):]
                elif name.startswith('base_model.model.'):
                    name = name[len('base_model.model.'):]
                converted_weights.append((name, tensor))
            # TODO: FP8 support
            if converted_weights:
                self.model_runner.model.load_weights(converted_weights)
    
    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for IPC communication."""
        if not hasattr(self, 'device_uuid') or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/twinkle-ipc-{self.device_uuid}.sock"
