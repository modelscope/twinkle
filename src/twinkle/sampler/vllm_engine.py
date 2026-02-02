# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union

import torch

from twinkle import remote_class, remote_function
from .base_engine import BaseSamplerEngine
from .types import StopReason, SamplingParams, SampleResponse, SampledSequence

import inspect

logger = logging.getLogger(__name__)





def get_vllm_max_lora_rank(lora_rank: int) -> int:
    """Get the nearest allowed vLLM LoRA rank."""
    from typing import get_args
    try:
        from vllm.config.lora import MaxLoRARanks
        allowed_ranks = sorted(get_args(MaxLoRARanks))
        for rank in allowed_ranks:
            if lora_rank <= rank:
                return rank
        return allowed_ranks[-1]
    except ImportError:
        # Fallback for older vLLM versions
        return lora_rank

@dataclass
class LoRAAdapter:
    """Represents a loaded LoRA adapter."""
    lora_int_id: int
    lora_name: str
    lora_path: str
    peft_config: Dict[str, Any]
    user_id: str
    created_at: float
    
    def to_lora_request(self):
        """Convert to vLLM LoRARequest."""
        from vllm.lora.request import LoRARequest
        return LoRARequest(
            lora_name=self.lora_name,
            lora_int_id=self.lora_int_id,
            lora_path=self.lora_path,
        )


class LoRAAdapterManager:
    def __init__(self, model_id: str, max_loras: int = 64):
        self.model_id = model_id
        self.max_loras = max_loras
        
        # user_id -> LoRAAdapter
        self._adapters: Dict[str, LoRAAdapter] = {}
        # lora_int_id -> user_id (for reverse lookup)
        self._id_to_user: Dict[int, str] = {}
        # Counter for generating unique lora_int_ids
        self._next_lora_id = 1
    
    def _generate_lora_id(self) -> int:
        """Generate a unique LoRA int ID."""
        lora_id = self._next_lora_id
        self._next_lora_id += 1
        # Wrap around if needed (vLLM has limits)
        if self._next_lora_id > 10000:
            self._next_lora_id = 1
        return lora_id
    
    def get_uri(self, user_id: str, version: Optional[int] = None) -> str:
        """
        Generate URI for a user's adapter.
        
        Args:
            user_id: User identifier
            version: Optional version number. If None, uses current adapter's lora_int_id
        
        Returns:
            URI like: twinkle://{model_id}/lora/{user_id}/{version}
        """
        if version is None:
            adapter = self._adapters.get(user_id)
            version = adapter.lora_int_id if adapter else 0
        return f"twinkle://{self.model_id}/lora/{user_id}/{version}"
    
    def parse_uri(self, uri: str) -> Optional[str]:
        """
        Parse user_id from URI. Returns None if invalid.
        """
        prefix = f"twinkle://{self.model_id}/lora/"
        if uri.startswith(prefix):
            remainder = uri[len(prefix):]
            # Handle version suffix: user_id/version or just user_id
            parts = remainder.split("/")
            return parts[0] if parts else None
        return None
    
    def get_adapter(self, user_id: str) -> Optional[LoRAAdapter]:
        """Get adapter for a user."""
        return self._adapters.get(user_id)
    
    def get_adapter_by_uri(self, uri: str) -> Optional[LoRAAdapter]:
        """Get adapter by URI."""
        user_id = self.parse_uri(uri)
        if user_id:
            return self.get_adapter(user_id)
        return None
    
    def add_adapter(
        self,
        user_id: str,
        peft_config: Dict[str, Any],
        lora_path: str,
    ) -> Tuple[LoRAAdapter, Optional[LoRAAdapter]]:
        """
        Add or replace adapter for a user.
        
        Returns:
            (new_adapter, old_adapter_to_remove)
        """
        old_adapter = self._adapters.get(user_id)
        
        # Clean up old adapter tracking
        if old_adapter:
            del self._id_to_user[old_adapter.lora_int_id]
        
        # Create new adapter
        lora_id = self._generate_lora_id()
        new_adapter = LoRAAdapter(
            lora_int_id=lora_id,
            lora_name=f"lora_{user_id}_{lora_id}",
            lora_path=lora_path,
            peft_config=peft_config,
            user_id=user_id,
            created_at=time.time(),
        )
        
        self._adapters[user_id] = new_adapter
        self._id_to_user[lora_id] = user_id
        
        return new_adapter, old_adapter
    
    def remove_adapter(self, user_id: str) -> Optional[LoRAAdapter]:
        """Remove adapter for a user."""
        adapter = self._adapters.pop(user_id, None)
        if adapter:
            self._id_to_user.pop(adapter.lora_int_id, None)
        return adapter
    
    def list_adapters(self) -> List[LoRAAdapter]:
        """List all adapters."""
        return list(self._adapters.values())

@remote_class()
class VLLMEngine(BaseSamplerEngine):
    """
    A vLLM-based inference engine for RL training.
    
    This engine uses vLLM v1's AsyncLLM and supports:
    - Tinker-compatible sample() API with logprobs
    - Multi-tenant LoRA adapters for client-server mode
    - Weight synchronization via load_weights (colocated) or CUDA IPC
    - Sleep/wake_up for GPU memory management in colocated training
    
    Deployment scenarios:
    1. Standalone server (client-server): Multi-tenant, LoRA adapters indexed by URI
    2. Colocated with training (Ray): Single-tenant, weight sync via load_weights
    """
    
    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enable_lora: bool = False,
        max_loras: int = 64,
        max_lora_rank: int = 64,
        enable_sleep_mode: bool = False,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        load_format: str = "auto",
        logprobs_mode: Optional[str] = None,
        **kwargs,
    ):
        from twinkle.hub import HubOperation
        model_id = HubOperation.download_model(model_id)
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.enable_sleep_mode = enable_sleep_mode
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.quantization = quantization
        self.load_format = load_format
        self.logprobs_mode = logprobs_mode
        self.engine_kwargs = kwargs or {}
        
        # LoRA adapter manager for multi-tenant mode
        self.adapter_manager = LoRAAdapterManager(model_id, max_loras) if enable_lora else None
        
        # Temp directory for LoRA weights (required by vLLM)
        self._lora_weights_dir = os.path.join("/tmp/twinkle_lora", hashlib.md5(model_id.encode()).hexdigest())
        os.makedirs(self._lora_weights_dir, exist_ok=True)
        
        # Initialize engine
        self.engine = self._create_engine()
        
        # Tokenizer is lazy loaded via get_tokenizer()
        self._tokenizer = None
    
    def _create_engine(self):
        """Create and return the vLLM engine."""
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        
        # Build engine config
        engine_config = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "dtype": self.dtype,
            "load_format": self.load_format,
            "distributed_executor_backend": "mp",
            "disable_log_stats": True,
        }
        
        if self.max_model_len is not None:
            engine_config["max_model_len"] = self.max_model_len
        
        if self.quantization is not None:
            engine_config["quantization"] = self.quantization
        
        if self.enable_prefix_caching:
            engine_config["enable_prefix_caching"] = True
        
        if self.enable_sleep_mode:
            engine_config["enable_sleep_mode"] = True
        
        if self.logprobs_mode is not None:
            engine_config["logprobs_mode"] = self.logprobs_mode
        
        if self.enable_lora:
            engine_config["enable_lora"] = True
            engine_config["max_loras"] = self.max_loras
            engine_config["max_lora_rank"] = get_vllm_max_lora_rank(self.max_lora_rank)
        
        # Enable worker extension for weight synchronization
        engine_config["worker_extension_cls"] = (
            "twinkle.sampler.vllm_worker_extension.TwinkleWorkerExtension"
        )
        
        engine_config.update(self.engine_kwargs)
        valid_args = inspect.signature(AsyncEngineArgs).parameters.keys()
        filtered_engine_config = {k: v for k, v in engine_config.items() if k in valid_args}
        invalid_args = set(engine_config.keys()) - set(valid_args)
        if invalid_args:
            logger.warning(f"VLLMEngine: Filtered out invalid arguments: {invalid_args}")
        # Create engine using vLLM v1 API
        engine_args = AsyncEngineArgs(**filtered_engine_config)
        vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
        
        engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        
        logger.info(f"VLLMEngine initialized: model={self.model_id}")
        return engine
    
    async def get_tokenizer(self):
        """Get the tokenizer asynchronously."""
        if self._tokenizer is None:
            self._tokenizer = await self.engine.get_tokenizer()
        return self._tokenizer
    
    # =========================================================================
    # Core Sampling API
    # =========================================================================
    
    @remote_function()
    async def sample(
        self,
        prompt_token_ids: List[int],
        sampling_params: Union[SamplingParams, Dict[str, Any]],
        num_samples: int = 1,
        logprobs: bool = True,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        adapter_uri: Optional[str] = None,
        request_id: Optional[str] = None,
        priority: int = 0,
        *,
        images: Optional[List[Any]] = None,
        videos: Optional[List[Any]] = None,
    ):
        """
        Sample completions from the model.
        
        This is the core API aligned with tinker's sampling interface.
        
        Args:
            prompt_token_ids: Input token IDs.
            sampling_params: Sampling parameters (tinker.types.SamplingParams or dict).
            num_samples: Number of samples to generate.
            logprobs: Whether to return log probabilities for generated tokens.
            include_prompt_logprobs: Whether to compute logprobs on prompt tokens.
            topk_prompt_logprobs: If > 0, returns top-k logprobs for each prompt token.
            adapter_uri: URI of LoRA adapter to use (for multi-tenant mode).
                         Format: twinkle://{model_id}/lora/{user_id}
            request_id: Optional request ID for tracking.
            priority: Request priority (higher = more urgent).
            images: Optional list of images for multimodal models.
                    Can be PIL.Image, file paths, URLs, or bytes.
            videos: Optional list of videos for multimodal models.
                    Can be file paths or list of frames.
            
        Returns:
            tinker.types.SampleResponse containing sequences and optionally prompt_logprobs.
        """
        from vllm.inputs import TokensPrompt
        
        # Convert to vLLM params
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams.from_dict(sampling_params)
        prompt_logprobs_k = topk_prompt_logprobs if topk_prompt_logprobs > 0 else (1 if include_prompt_logprobs else 0)
        vllm_params = sampling_params.to_vllm(logprobs=logprobs, prompt_logprobs=prompt_logprobs_k)
        
        # Build request
        if request_id is None:
            request_id = uuid.uuid4().hex
        
        # Build multi_modal_data if images or videos provided
        multi_modal_data = {}
        if images:
            multi_modal_data['image'] = images
        if videos:
            multi_modal_data['video'] = videos
        
        # Build prompt (with or without multimodal data)
        if multi_modal_data:
            prompt = TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data=multi_modal_data,
            )
        else:
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        
        # Build LoRA request if adapter_uri provided
        lora_request = None
        if adapter_uri and self.adapter_manager:
            adapter = self.adapter_manager.get_adapter_by_uri(adapter_uri)
            if adapter:
                # Check if adapter is loaded in engine
                loaded_loras = await self.engine.list_loras()
                if adapter.lora_int_id in loaded_loras:
                    lora_request = adapter.to_lora_request()
                else:
                    logger.warning(f"Adapter {adapter_uri} not loaded in engine, sampling without adapter")
        
        # Generate
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=vllm_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )
        
        # Get final result
        result = None
        async for output in generator:
            result = output
        
        if result is None:
            raise RuntimeError("Sampling did not produce a result")
        
        # Extract sequences
        sequences = []
        for output in result.outputs:
            token_ids = list(output.token_ids)
            
            # Extract logprobs
            seq_logprobs = None
            if output.logprobs is not None:
                seq_logprobs = []
                for i, lp in enumerate(output.logprobs):
                    if i < len(token_ids) and token_ids[i] in lp:
                        seq_logprobs.append(lp[token_ids[i]].logprob)
            
            # Map finish_reason to StopReason
            stop_reason: StopReason = "length"
            if output.finish_reason in ("stop", "eos_token"):
                stop_reason = "stop"
            
            sequences.append(SampledSequence(
                stop_reason=stop_reason,
                tokens=token_ids,
                logprobs=seq_logprobs,
            ))
        
        # Extract prompt logprobs if requested
        result_prompt_logprobs = None
        result_topk_prompt_logprobs = None
        if prompt_logprobs_k > 0 and result.prompt_logprobs is not None:
            result_prompt_logprobs = []
            result_topk_prompt_logprobs = []
            
            for i, lp_dict in enumerate(result.prompt_logprobs):
                if lp_dict is None:
                    result_prompt_logprobs.append(None)
                    result_topk_prompt_logprobs.append(None)
                    continue
                
                # Get logprob for the actual token
                if i < len(prompt_token_ids):
                    token_id = prompt_token_ids[i]
                    if token_id in lp_dict:
                        lp_obj = lp_dict[token_id]
                        result_prompt_logprobs.append(
                            lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj
                        )
                    else:
                        result_prompt_logprobs.append(None)
                else:
                    result_prompt_logprobs.append(None)
                
                # Get top-k logprobs
                sorted_items = sorted(
                    lp_dict.items(),
                    key=lambda x: -(x[1].logprob if hasattr(x[1], 'logprob') else x[1])
                )[:prompt_logprobs_k]
                result_topk_prompt_logprobs.append([
                    (tid, lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj)
                    for tid, lp_obj in sorted_items
                ])
        
        return SampleResponse(
            sequences=sequences,
            prompt_logprobs=result_prompt_logprobs,
            topk_prompt_logprobs=result_topk_prompt_logprobs,
        )

    @remote_function()
    async def save_weights_for_sampler(
        self,
        user_id: str,
        weights: Dict[str, torch.Tensor],
        peft_config: Dict[str, Any],
        *,
        ttl_seconds: int = 604800,  # 7 days
    ) -> str:
        """
        Save weights as a LoRA adapter for sampling (client-server mode).
        
        This is aligned with tinker's save_weights_for_sampler API.
        For multi-tenant scenarios:
        - Each user identified by user_id
        - Old adapter for the same user is automatically removed
        - Returns a URI that can be passed to sample()
        
        Args:
            user_id: Unique identifier for the user/training run.
            weights: LoRA weight tensors.
            peft_config: PEFT/LoRA configuration dict.
            ttl_seconds: Time-to-live for the adapter (not enforced yet).
            
        Returns:
            URI string: twinkle://{model_id}/lora/{user_id}
        """
        if not self.enable_lora or not self.adapter_manager:
            raise RuntimeError("LoRA not enabled. Set enable_lora=True.")
        
        # Save weights to disk (vLLM requires file path)
        adapter_dir = os.path.join(self._lora_weights_dir, user_id)
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Save adapter weights
        import safetensors.torch
        safetensors.torch.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))
        
        # Save adapter config
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(peft_config, f)
        
        # Register adapter (removes old one for same user)
        new_adapter, old_adapter = self.adapter_manager.add_adapter(
            user_id=user_id,
            peft_config=peft_config,
            lora_path=adapter_dir,
        )
        
        # Remove old adapter from engine if exists
        if old_adapter:
            try:
                loaded_loras = await self.engine.list_loras()
                if old_adapter.lora_int_id in loaded_loras:
                    await self.engine.collective_rpc(
                        method="remove_lora",
                        args=(old_adapter.lora_int_id,),
                    )
                    logger.info(f"Removed old adapter for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to remove old adapter: {e}")
        
        # Load new adapter into engine using TensorLoRARequest
        from twinkle.patch.vllm_lora_weights import TensorLoRARequest
        
        lora_request = TensorLoRARequest(
            lora_name=new_adapter.lora_name,
            lora_int_id=new_adapter.lora_int_id,
            lora_path=new_adapter.lora_path,
            peft_config=peft_config,
            lora_tensors=weights,
        )
        
        await self.engine.collective_rpc(
            method="add_lora",
            args=(lora_request,),
        )
        
        logger.info(f"Loaded LoRA adapter for user {user_id}: {len(weights)} tensors")
        
        return self.adapter_manager.get_uri(user_id)
    
    async def remove_adapter(self, user_id: str) -> bool:
        """
        Remove a LoRA adapter for a user.
        
        Args:
            user_id: User identifier.
            
        Returns:
            True if adapter was removed, False if not found.
        """
        if not self.adapter_manager:
            return False
        
        adapter = self.adapter_manager.remove_adapter(user_id)
        if adapter:
            try:
                await self.engine.collective_rpc(
                    method="remove_lora",
                    args=(adapter.lora_int_id,),
                )
            except Exception as e:
                logger.warning(f"Failed to remove adapter from engine: {e}")
            return True
        return False

    async def sleep(self, level: int = 2) -> None:
        """
        Offload weights and/or KV cache from GPU memory.
        
        Used in colocated mode to free GPU memory for training.
        
        Args:
            level: Sleep level.
                1 = offload KV cache only
                2 = offload KV cache and weights
        """
        if not self.enable_sleep_mode:
            logger.warning("sleep_mode not enabled, skipping sleep")
            return
        
        await self.engine.sleep(level=level)
        logger.debug(f"Engine sleeping at level {level}")
    
    async def wake_up(self, tags: Optional[List[str]] = None, reload_weights: bool = False) -> None:
        """
        Resume weights and/or KV cache to GPU memory.
        
        Used in colocated mode before inference.
        
        Args:
            tags: What to resume. Options: ['weights', 'kv_cache'].
                  If None, resumes both.
            reload_weights: If True and level 2 sleep was used (weights discarded),
                  reload weights from disk via collective_rpc("reload_weights").

        """
        if not self.enable_sleep_mode:
            logger.warning("sleep_mode not enabled, skipping wake_up")
            return
        
        if tags is None:
            tags = ["weights", "kv_cache"]
        
        await self.engine.wake_up(tags=tags)
        
        if reload_weights and "weights" in tags:
            try:
                await self.engine.collective_rpc("reload_weights")
                logger.debug("Weights reloaded after wake_up")
            except Exception as e:
                logger.warning(f"Failed to reload weights: {e}")
        
        await self.clear_kv_cache()
        
        logger.debug(f"Engine waking up with tags: {tags}")

    async def clear_kv_cache(self) -> None:
        """Clear the KV cache (prefix cache)."""
        if hasattr(self.engine, 'reset_prefix_cache'):
            await self.engine.reset_prefix_cache()
        elif hasattr(self.engine, 'reset_mm_cache'):
            await self.engine.reset_mm_cache() # Do we need this?

    async def abort_request(self, request_id: str) -> None:
        """Abort a specific request."""
        await self.engine.abort(request_id)
    
    async def list_adapters(self) -> List[Dict[str, Any]]:
        """List all loaded LoRA adapters."""
        if not self.adapter_manager:
            return []
        
        adapters = self.adapter_manager.list_adapters()
        return [
            {
                "user_id": a.user_id,
                "uri": self.adapter_manager.get_uri(a.user_id),
                "lora_int_id": a.lora_int_id,
                "created_at": a.created_at,
            }
            for a in adapters
        ]
