# Copyright (c) ModelScope Contributors. All rights reserved.
"""CUDA IPC Weight Loader for Hybrid mode.

This loader synchronizes weights from training model to vLLM sampler
using CUDA IPC for efficient GPU-to-GPU transfer within the same machine.

Architecture:
    Training Model (main process)
           │
           │ get_hf_state_dict() -> Generator[(name, tensor)]
           ▼
    IPCWeightLoader
           │
           │ ZMQ + CUDA IPC (bucket-based)
           ▼
    vLLM Worker (subprocess)
           │
           └── TwinkleWorkerExtension.update_weights_from_ipc()

Supported Modes:
    - HYBRID: Model and Sampler in same process, vLLM in subprocess (same GPU)
    - COLOCATE: Model and Sampler in different processes (same GPU) - needs ray handle support

Limitations:
    - CUDA IPC only works on the same physical GPU
    - For STANDALONE mode (different GPUs), use NCCLWeightLoader instead
"""
import concurrent.futures
import gc
import logging
import os
import uuid
from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union

import torch
from torch.multiprocessing.reductions import reduce_tensor

from twinkle.model.base import TwinkleModel
from twinkle.sampler.base import Sampler
from twinkle.utils.framework import Torch
from .base import WeightLoader

logger = logging.getLogger(__name__)


def get_device_uuid(device_id: int) -> str:
    """Get unique device identifier."""
    try:
        from vllm.platforms import current_platform
        # For NPU, handle ASCEND_RT_VISIBLE_DEVICES
        if Torch.is_npu_available():
            npu_visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
            if device_id < len(npu_visible_devices):
                return "NPU-" + npu_visible_devices[device_id]
        return current_platform.get_device_uuid(device_id)
    except Exception:
        # Fallback to random UUID if vllm not available
        return uuid.uuid4().hex[:16]


def is_ipc_supported() -> bool:
    """Check if CUDA/NPU IPC is supported."""
    if Torch.is_gpu_available():
        return True
    if Torch.is_npu_available():
        # NPU IPC requires specific versions
        # Ascend HDK >= 25.3.rc1 and CANN >= 8.3.RC1
        # TODO: Add version check
        return True
    return False


class IPCWeightLoader(WeightLoader):
    """Weight loader using CUDA IPC for Hybrid mode.
    
    This loader is designed for scenarios where training model and sampler
    are in the same Ray Worker but vLLM runs in subprocess (spawn mode).
    
    Features:
    - CUDA IPC for zero-copy GPU memory sharing
    - Bucket-based streaming transfer (avoids OOM for large models)
    - Fallback to shared memory when CUDA IPC not supported
    
    Args:
        model: Training model instance (TransformersModel/MegatronModel)
        sampler: Sampler instance (VLLMSampler)
        bucket_size_mb: Size of transfer bucket in MB (default: 512)
        use_shm: Force use shared memory instead of CUDA IPC
        dtype: Target dtype for weights (default: bfloat16)
    
    Note:
        - For Hybrid mode, model and sampler must be actual objects (not Ray handles)
        - For Colocate mode with Ray handles, additional handling is needed
        - This loader only supports same-GPU scenarios (CUDA IPC limitation)
        - For cross-GPU scenarios (STANDALONE mode), use NCCLWeightLoader
    
    Example:
        >>> model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B")
        >>> sampler = VLLMSampler(model_id="Qwen/Qwen2.5-0.5B", 
        ...                       engine_args={'load_format': 'dummy'})
        >>> loader = IPCWeightLoader(model, sampler)
        >>> loader.load_weights()  # Sync model weights to sampler
    """
    
    def __init__(
        self,
        model: TwinkleModel,
        sampler: Sampler,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.sampler = sampler
        self.bucket_size = bucket_size_mb << 20  # Convert to bytes
        self.use_shm = use_shm or not is_ipc_supported()
        self.dtype = dtype
        
        self._zmq_ctx = None
        self._device_uuid = None
        self.base_sync_done = False
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation."
            )
    
    @property
    def device_uuid(self) -> str:
        """Get or compute device UUID."""
        if self._device_uuid is None:
            device_id = Torch.get_current_device()
            if isinstance(device_id, str):
                device_id = 0
            self._device_uuid = get_device_uuid(device_id)
        return self._device_uuid
    
    @property
    def zmq_handle(self) -> str:
        """Get ZMQ IPC socket address."""
        return f"ipc:///tmp/twinkle-ipc-{self.device_uuid}.sock"
    
    def load_weights(self, adapter_name: str = '', peft_config: Optional[Dict] = None):
        """Sync weights from model to sampler via CUDA IPC.
        
        This is the main entry point for weight synchronization.
        
        Args:
            adapter_name: Name of the adapter (for LoRA)
            peft_config: PEFT config for LoRA mode. When provided with base_sync_done=True,
                        only LoRA weights are synced (assuming base model is already loaded in vLLM).
        """
        import zmq
        
        # Get weights iterator from training model
        # For TransformersModel: returns dict of {name: tensor}
        # For MegatronModel: get_hf_state_dict() returns Generator[(name, tensor)]
        # Using iterator directly avoids OOM for large models
        weights_source = self._get_weights_iterator(adapter_name)
        
        logger.info("Starting CUDA IPC weight sync...")
        
        # Step 1: Setup ZMQ sender FIRST (bind before worker connects)
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REQ)
        socket.bind(self.zmq_handle)
        logger.debug(f"ZMQ socket bound to {self.zmq_handle}")
        
        # Step 2: Trigger vLLM worker to start receiving (non-blocking)
        # Worker will connect to ZMQ and wait for data
        receiver_future = self._trigger_receiver(peft_config, base_sync_done=self.base_sync_done)
        
        # Give worker time to connect
        import time
        time.sleep(0.5)
        
        try:
            # Step 3: Create transfer buffer and send handle
            buffer, shm = self._create_buffer(socket)
            
            # Step 4: Send weights in buckets (streaming, no full list)
            count = self._send_weights_in_buckets(socket, buffer, weights_source)
            
            # Step 5: Wait for receiver to complete processing
            # This ensures the collective_rpc is fully done before returning
            try:
                receiver_future.result(timeout=60)  # 60s timeout
            except Exception as e:
                logger.warning(f"Receiver future completed with: {e}")
            
            # Step 6: Clear KV cache after weight update
            # This is necessary because the model weights have changed
            self._clear_kv_cache()
            
            logger.info(f"CUDA IPC weight sync completed: {count} tensors")
            
        finally:
            # Cleanup
            socket.close()
            if buffer is not None:
                del buffer
            if shm is not None:
                shm.close()
                shm.unlink()
            gc.collect()
            Torch.ipc_collect()
            Torch.empty_cache()
    
    def _get_weights_iterator(self, adapter_name: str = '') -> Iterable[Tuple[str, torch.Tensor]]:
        """Get weights iterator from model.
        
        This method handles both TransformersModel (dict) and MegatronModel (generator).
        For MegatronModel, it uses get_hf_state_dict() which returns a generator
        that converts Megatron format to HF format on-the-fly.
        
        Args:
            adapter_name: Name of the adapter (for LoRA)
            
        Returns:
            Iterable of (name, tensor) pairs
        """
        # Check if model has get_hf_state_dict (MegatronModel)
        if hasattr(self.model, 'get_hf_state_dict'):
            # MegatronModel: returns generator that converts to HF format
            return self.model.get_hf_state_dict(adapter_name=adapter_name)
        else:
            # TransformersModel: returns dict
            state_dict = self.model.get_state_dict(adapter_name=adapter_name)
            if isinstance(state_dict, dict):
                return state_dict.items()
            else:
                # Already an iterator/generator
                return state_dict
    
    def _trigger_receiver(self, peft_config: Optional[Dict], base_sync_done: bool = False) -> "concurrent.futures.Future":
        """Trigger vLLM worker to start receiving weights.
        
        This calls the sampler's engine to invoke collective_rpc on all
        vLLM workers to start the update_weights_from_ipc method.
        
        IMPORTANT: This method triggers the receiver in a non-blocking way.
        The collective_rpc call starts the worker method but doesn't wait for
        it to complete, allowing the sender to start sending data.
        
        Args:
            peft_config: PEFT config for LoRA mode.
            base_sync_done: If True and peft_config provided, only sync LoRA weights.
                           If False and peft_config provided, sync base model weights first.
        
        Returns:
            Future that can be awaited after sending weights.
        """
        import asyncio
        
        # Access vLLM engine's collective_rpc through sampler
        # VLLMSampler -> VLLMEngine -> AsyncLLM -> collective_rpc
        engine = self.sampler.engine
        
        # Run the async trigger in the sampler's event loop
        async def _trigger():
            # collective_rpc call - worker will start receiving and wait on ZMQ socket
            return await engine.engine.collective_rpc(
                "update_weights_from_ipc",
                kwargs={
                    "peft_config": peft_config, 
                    "use_shm": self.use_shm,
                    "base_sync_done": base_sync_done,
                },
            )
        
        # Schedule the task without waiting for completion
        # This allows the sender to proceed immediately
        future = asyncio.run_coroutine_threadsafe(
            _trigger(), 
            self.sampler._async_loop
        )
        # Return future so caller can wait after sending weights
        return future
    
    def _create_buffer(self, socket) -> Tuple[torch.Tensor, Any]:
        """Create transfer buffer and send handle to receiver."""
        buffer = None
        shm = None
        
        if not self.use_shm:
            # CUDA IPC mode
            device = Torch.get_device(None)
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=device)
            handle = reduce_tensor(buffer)
            socket.send_pyobj(handle)
        else:
            # Shared memory mode
            from multiprocessing import shared_memory
            
            shm_name = f"twinkle_weights_{uuid.uuid4().hex}"
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.bucket_size)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            socket.send_pyobj({"name": shm_name, "size": self.bucket_size})
        
        socket.recv()  # Wait for receiver ready
        return buffer, shm
    
    def _send_weights_in_buckets(
        self,
        socket,
        buffer: torch.Tensor,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> int:
        """Send weights in buckets via streaming.
        
        This method processes weights one by one without loading all into memory,
        which is critical for large models to avoid OOM.
        
        Args:
            socket: ZMQ socket for communication
            buffer: CUDA IPC buffer for weight transfer
            weights: Iterable of (name, tensor) pairs
            
        Returns:
            Number of tensors sent
        """
        offset = 0
        bucket_meta = {}
        count = 0
        
        for name, weight in weights:
            # Convert DTensor to local tensor if needed
            weight = Torch.to_local_tensor(weight)
            
            # Convert to target dtype
            weight = weight.to(self.dtype, non_blocking=True)
            
            # Check if bucket is full
            if offset + weight.nbytes > self.bucket_size:
                Torch.synchronize()
                socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                socket.recv()
                bucket_meta = {}
                offset = 0
            
            # Validate weight fits in bucket
            if weight.nbytes > self.bucket_size:
                raise ValueError(
                    f"Weight '{name}' ({weight.shape}, {weight.dtype}) is too large "
                    f"({weight.nbytes / 1e6:.1f}MB) for bucket ({self.bucket_size / 1e6:.1f}MB). "
                    f"Increase bucket_size_mb."
                )
            
            # Add weight to bucket
            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            buffer[offset:offset + weight.nbytes].copy_(
                weight.view(-1).view(torch.uint8), non_blocking=True
            )
            offset += weight.nbytes
            count += 1
        
        # Send final bucket
        Torch.synchronize()
        socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
        socket.recv()
        
        return count
    
    def _clear_kv_cache(self) -> None:
        """Clear KV cache after weight update.
        
        This is necessary because the model weights have changed and
        any cached KV pairs are now invalid.
        """
        import asyncio
        
        engine = self.sampler.engine
        
        async def _clear():
            await engine.clear_kv_cache()
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                _clear(), 
                self.sampler._async_loop
            )
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Failed to clear KV cache: {e}")
    
    def __call__(
        self,
        model: TwinkleModel = None,
        sampler: Sampler = None,
        adapter_name: str = '',
        base_sync_done: Optional[bool] = None,
    ):
        """Callable interface for WeightLoader protocol."""
        if model is not None:
            self.model = model
        if sampler is not None:
            self.sampler = sampler
        if base_sync_done is not None:
            self.base_sync_done = base_sync_done
        self.load_weights(adapter_name=adapter_name)
