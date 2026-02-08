# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Base sampler engine abstract class.

This module defines the interface that all sampler engines must implement.
Engines are the low-level components that handle token-based inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from twinkle.data_format import SamplingParams, SampleResponse


class BaseSamplerEngine(ABC):

    @abstractmethod
    async def sample(
        self,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        *,
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        request_id: Optional[str] = None,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
        **kwargs,
    ) -> SampleResponse:
        """
        Sample completions from the model.
        
        Args:
            prompt_token_ids: Input token IDs.
            sampling_params: Sampling parameters.
            num_samples: Number of samples to generate.
            logprobs: Whether to return log probabilities for generated tokens.
            include_prompt_logprobs: Whether to compute logprobs on prompt tokens.
            topk_prompt_logprobs: If > 0, returns top-k logprobs for each prompt token.
            adapter_uri: URI of LoRA adapter to use (for multi-tenant mode).
            request_id: Optional request ID for tracking.
            images: Optional list of images for multimodal models.
                    Can be PIL.Image, file paths, URLs, or bytes.
                    VLLMEngine passes these directly to vLLM.
                    TransformersEngine requires pre-processed inputs via extra_model_inputs.
            videos: Optional list of videos for multimodal models.
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
        user_id: str,
        weights: Dict[str, torch.Tensor],
        peft_config: Dict[str, Any],
        **kwargs,
    ) -> str:
        """
        Save weights as a LoRA adapter for sampling (client-server mode).
        
        Args:
            user_id: Unique identifier for the user/training run.
            weights: LoRA weight tensors.
            peft_config: PEFT/LoRA configuration dict.
            
        Returns:
            URI string for the adapter.
        """
        raise NotImplementedError("save_weights_for_sampler not implemented")
    
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
