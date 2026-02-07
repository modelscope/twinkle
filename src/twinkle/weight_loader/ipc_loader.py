# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight Loader for Hybrid mode.

Synchronizes weights from training model to vLLM sampler in Hybrid
deployment, where model and sampler live in the same Ray Worker process
but vLLM runs in a subprocess.

The weights are collected from the model and passed to vLLM via
``VLLMEngine.update_weights()`` → ``collective_rpc`` →
``TwinkleWorkerExtension.load_synced_weights()`` in the worker subprocess.

Limitations:
    - For STANDALONE mode (different GPUs), use CheckpointEngineManager instead.
"""
import asyncio
import gc
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from twinkle.model.base import TwinkleModel
from twinkle.sampler.base import Sampler
from twinkle.utils.framework import Torch
from .base import WeightLoader

logger = logging.getLogger(__name__)


class IPCWeightLoader(WeightLoader):
    """Weight loader for Hybrid mode.

    Collects model weights and transfers them to the vLLM subprocess via
    ``VLLMEngine.update_weights()`` (which uses ``collective_rpc``).

    Args:
        model: Training model instance (TransformersModel/MegatronModel).
        sampler: Sampler instance (VLLMSampler).
        dtype: Target dtype for weights (default: bfloat16).

    Example:
        >>> model = TransformersModel(model_id="Qwen/Qwen2.5-0.5B")
        >>> sampler = VLLMSampler(model_id="Qwen/Qwen2.5-0.5B",
        ...                       engine_args={'load_format': 'dummy'})
        >>> loader = IPCWeightLoader(model, sampler)
        >>> loader.load_weights()
    """

    def __init__(
        self,
        model: TwinkleModel,
        sampler: Sampler,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        self.model = model
        self.sampler = sampler
        self.dtype = dtype
        self.base_sync_done = False

    def load_weights(self, adapter_name: str = '', peft_config: Optional[Dict] = None):
        """Sync weights from model to sampler.

        Args:
            adapter_name: Name of the adapter (for LoRA, reserved).
            peft_config: PEFT config dict for LoRA adapter loading.
        """
        import time

        start_time = time.time()

        # Collect weights from training model
        weights = {}
        for name, tensor in self._get_weights_iterator(adapter_name):
            tensor = Torch.to_local_tensor(tensor)
            weights[name] = tensor.to(self.dtype, non_blocking=True)
        Torch.synchronize()

        # Transfer to vLLM subprocess via collective_rpc
        engine = self.sampler.engine
        future = asyncio.run_coroutine_threadsafe(
            engine.update_weights(
                weights,
                peft_config=peft_config,
                base_sync_done=self.base_sync_done,
            ),
            self.sampler._async_loop,
        )
        future.result(timeout=120)

        # Clear KV cache since model weights changed
        self._clear_kv_cache()

        del weights
        gc.collect()
        Torch.empty_cache()

        elapsed = time.time() - start_time
        logger.info(f"Weight sync completed in {elapsed:.2f}s")

    def _get_weights_iterator(self, adapter_name: str = '') -> Iterable[Tuple[str, torch.Tensor]]:
        """Get weights iterator from the local model object.

        Supports TransformersModel (state_dict → dict) and
        MegatronModel (get_hf_state_dict → generator).
        """
        if hasattr(self.model, 'get_hf_state_dict'):
            return self.model.get_hf_state_dict()
        else:
            return self.model.state_dict()

    def _clear_kv_cache(self) -> None:
        """Clear KV cache after weight update."""
        engine = self.sampler.engine
        try:
            future = asyncio.run_coroutine_threadsafe(
                engine.clear_kv_cache(),
                self.sampler._async_loop,
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
