# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Base sampler engine abstract class.

This module defines the interface that all sampler engines must implement.
Engines are the low-level components that handle token-based inference.
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import SampleResponse, SamplingParams


class BaseSamplerEngine(ABC):

    @abstractmethod
    async def sample(
        self,
        prompt: Union[List[int], str],
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        lora_request: Optional[Any] = None,
        request_id: Optional[str] = None,
        priority: int = 0,
        *,
        multi_modal_data: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        disable_lora: bool = False,
        **kwargs,
    ) -> SampleResponse:
        """
        Sample completions from the model.

        Args:
            prompt: Input token IDs or text.
            sampling_params: Sampling parameters.
            lora_request: Optional vLLM LoRA request.
            request_id: Optional request ID for tracking.
            priority: Request scheduling priority.
            multi_modal_data: Image/video data in the inference engine's format.
            mm_processor_kwargs: Multimodal processor overrides.
            disable_lora: Sample from base model weights.
            **kwargs: Additional engine-specific arguments.

        Returns:
            SampleResponse containing sequences and optionally prompt_logprobs.
        """
        pass

    @abstractmethod
    async def get_tokenizer(self):
        """Get the tokenizer."""
        pass

    async def update_weights(
        self,
        weights: Dict[str, torch.Tensor],
        adapter_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Update model weights.

        Args:
            weights: Dict of (name, tensor) pairs.
            adapter_name: If provided, update LoRA adapter weights instead of base model.
        """
        pass

    async def save_weights_for_sampler(
        self,
        weights: Dict[str, torch.Tensor],
        peft_config: Dict[str, Any],
        **kwargs,
    ) -> str:
        """
        Save weights as a LoRA adapter for sampling (client-server mode).

        Args:
            weights: LoRA weight tensors.
            peft_config: PEFT/LoRA configuration dict.

        Returns:
            URI string for the adapter.
        """
        raise NotImplementedError('save_weights_for_sampler not implemented')

    async def sleep(self, **kwargs) -> None:
        """
        Offload weights from GPU memory (for colocated training).
        """
        pass

    async def wake_up(self, **kwargs) -> None:
        """
        Reload weights to GPU memory (for colocated training).
        """
        pass
