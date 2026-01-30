# Copyright (c) ModelScope Contributors. All rights reserved.
"""
SamplingClient for Twinkle - compatible with Tinker's SamplingClient API.

This module provides a client for text generation and inference from trained or base models.
It is designed to be compatible with Tinker's SamplingClient API.

Usage:
    # Create via ServiceClient
    service_client = tinker.ServiceClient(base_url="http://localhost:8000/api/v1")
    sampling_client = service_client.create_sampling_client(model_path="twinkle://model/lora/user1")
    
    # Sample
    response = sampling_client.sample(
        prompt=model_input,
        num_samples=4,
        sampling_params=tinker.types.SamplingParams(max_tokens=256),
    ).result()
    
    # Or use async
    response = await sampling_client.sample_async(...)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future as ConcurrentFuture
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from tinker import types
    from transformers import PreTrainedTokenizer


class SamplingClient:
    """
    Client for text generation and inference from trained or base models.
    
    This is compatible with Tinker's SamplingClient API.
    
    Args:
        base_url: Base URL of the Twinkle/Tinker server.
        model_path: Path to saved model weights (twinkle:// or tinker:// URI).
        base_model: Name of base model to use for inference.
        session: HTTP session for making requests.
    """
    
    def __init__(
        self,
        base_url: str,
        model_path: Optional[str] = None,
        base_model: Optional[str] = None,
        session: Optional[Any] = None,
        default_headers: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_path = model_path
        self.base_model = base_model
        self._session = session
        self._default_headers = default_headers or {}
        self._tokenizer = None
    
    def _get_session(self):
        """Get or create an HTTP session."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(
                timeout=None,
                headers=self._default_headers,
            )
        return self._session
    
    async def _request(self, endpoint: str, data: dict) -> dict:
        """Make an async request to the server."""
        session = self._get_session()
        url = f"{self.base_url}/{endpoint}"
        response = await session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def _wait_for_future(self, request_id: str) -> dict:
        """Poll for a future result."""
        while True:
            result = await self._request("retrieve_future", {"request_id": request_id})
            if result.get("error"):
                raise RuntimeError(result["error"])
            if result is not None:
                return result
            await asyncio.sleep(0.1)
    
    def sample(
        self,
        prompt: "types.ModelInput",
        num_samples: int,
        sampling_params: "types.SamplingParams",
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> ConcurrentFuture["types.SampleResponse"]:
        """
        Generate text completions from the model.
        
        Args:
            prompt: The input tokens as ModelInput.
            num_samples: Number of independent samples to generate.
            sampling_params: Parameters controlling generation.
            include_prompt_logprobs: Whether to include log probabilities for prompt tokens.
            topk_prompt_logprobs: Number of top token log probabilities to return per position.
            
        Returns:
            A Future containing the SampleResponse with generated text.
        """
        future = ConcurrentFuture()
        
        async def _do_sample():
            try:
                result = await self.sample_async(
                    prompt=prompt,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs,
                )
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        # Schedule the async task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_sample())
        except RuntimeError:
            # No running loop, run in a new thread
            import threading
            def run():
                asyncio.run(_do_sample())
            threading.Thread(target=run, daemon=True).start()
        
        return future
    
    async def sample_async(
        self,
        prompt: "types.ModelInput",
        num_samples: int,
        sampling_params: "types.SamplingParams",
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> "types.SampleResponse":
        """
        Async version of sample.
        
        Args:
            prompt: The input tokens as ModelInput.
            num_samples: Number of independent samples to generate.
            sampling_params: Parameters controlling generation.
            include_prompt_logprobs: Whether to include log probabilities for prompt tokens.
            topk_prompt_logprobs: Number of top token log probabilities to return per position.
            
        Returns:
            SampleResponse with generated text.
        """
        from tinker import types
        
        # Build request
        request_data = {
            "prompt": prompt.model_dump() if hasattr(prompt, 'model_dump') else prompt,
            "num_samples": num_samples,
            "sampling_params": sampling_params.model_dump() if hasattr(sampling_params, 'model_dump') else sampling_params,
            "prompt_logprobs": include_prompt_logprobs,
            "topk_prompt_logprobs": topk_prompt_logprobs,
        }
        
        if self.model_path:
            request_data["model_path"] = self.model_path
        if self.base_model:
            request_data["base_model"] = self.base_model
        
        # Submit request
        future_response = await self._request("asample", request_data)
        request_id = future_response.get("request_id")
        
        if request_id is None:
            # Direct response (no future)
            return types.SampleResponse(**future_response)
        
        # Wait for result
        result = await self._wait_for_future(request_id)
        return types.SampleResponse(**result)
    
    def compute_logprobs(
        self,
        prompt: "types.ModelInput",
    ) -> ConcurrentFuture[List[Optional[float]]]:
        """
        Compute log probabilities for prompt tokens.
        
        Args:
            prompt: The input tokens as ModelInput.
            
        Returns:
            A Future containing a list of log probabilities for each token in the prompt.
        """
        future = ConcurrentFuture()
        
        async def _do_compute():
            try:
                result = await self.compute_logprobs_async(prompt)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_do_compute())
        except RuntimeError:
            import threading
            def run():
                asyncio.run(_do_compute())
            threading.Thread(target=run, daemon=True).start()
        
        return future
    
    async def compute_logprobs_async(
        self,
        prompt: "types.ModelInput",
    ) -> List[Optional[float]]:
        """
        Async version of compute_logprobs.
        
        Args:
            prompt: The input tokens as ModelInput.
            
        Returns:
            List of log probabilities for each token in the prompt.
        """
        from tinker import types
        
        # Use sample with max_tokens=1 and include_prompt_logprobs=True
        response = await self.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
        )
        
        return response.prompt_logprobs
    
    def get_tokenizer(self) -> "PreTrainedTokenizer":
        """
        Get the tokenizer for the current model.
        
        Returns:
            PreTrainedTokenizer compatible with the model.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            model_name = self.base_model
            if model_name is None and self.model_path:
                # Try to extract model name from path
                # Format: twinkle://model_name/lora/user_id
                parts = self.model_path.replace("twinkle://", "").replace("tinker://", "").split("/")
                if parts:
                    model_name = parts[0]
            
            if model_name is None:
                raise ValueError("Cannot determine model name for tokenizer")
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        return self._tokenizer


def create_sampling_client(
    base_url: str,
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
    **kwargs,
) -> SamplingClient:
    """
    Create a SamplingClient for text generation.
    
    Args:
        base_url: Base URL of the Twinkle/Tinker server.
        model_path: Path to saved model weights (twinkle:// or tinker:// URI).
        base_model: Name of base model to use for inference.
        
    Returns:
        SamplingClient instance.
    """
    return SamplingClient(
        base_url=base_url,
        model_path=model_path,
        base_model=base_model,
        **kwargs,
    )
