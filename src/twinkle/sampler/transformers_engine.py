# Copyright (c) ModelScope Contributors. All rights reserved.
"""
TransformersEngine: A transformers-based inference engine.

Uses HuggingFace transformers model.generate() for text generation.
Slower than vLLM but more compatible and easier to debug.
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from twinkle import remote_class, remote_function
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from .base_engine import BaseSamplerEngine
from .types import SampleResponse, SampledSequence, SamplingParams

logger = logging.getLogger(__name__)


@remote_class()
class TransformersEngine(BaseSamplerEngine):
    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        model_kwargs: Optional[Dict[str, Any]] = None,
        model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = AutoModelForCausalLM,
    ):
        self._model_id = model_id
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self._model_kwargs = model_kwargs or {}
        
        # Load model and tokenizer
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **self._model_kwargs,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # LoRA adapter management
        self._adapters: Dict[str, Dict[str, Any]] = {}
        self._lora_weights_dir = os.path.join("/tmp/twinkle_lora", hashlib.md5(model_id.encode()).hexdigest())
        os.makedirs(self._lora_weights_dir, exist_ok=True)
        
        # Track current adapter
        self._current_adapter: Optional[str] = None
        
        logger.info(f"TransformersEngine initialized: model={model_id}")
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    async def get_tokenizer(self):
        return self.tokenizer
    
    def _convert_params(self, params: Optional[SamplingParams]) -> Dict[str, Any]:
        """Convert SamplingParams to transformers generate kwargs."""
        if params is None:
            params = SamplingParams()
        
        gen_kwargs = {
            "do_sample": params.temperature > 0,
            "temperature": max(params.temperature, 1e-7),
            "top_p": params.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if params.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = params.max_tokens
        else:
            gen_kwargs["max_new_tokens"] = 2048
        
        if params.seed is not None:
            torch.manual_seed(params.seed)
        
        if params.top_k > 0:
            gen_kwargs["top_k"] = params.top_k
        
        if params.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = params.repetition_penalty
        
        # Handle stop sequences
        if params.stop:
            if isinstance(params.stop, str):
                stop_token_ids = self.tokenizer.encode(params.stop, add_special_tokens=False)
                if stop_token_ids:
                    gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + stop_token_ids
            elif isinstance(params.stop, (list, tuple)):
                if params.stop and isinstance(params.stop[0], int):
                    gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + list(params.stop)
                else:
                    all_stop_ids = [self.tokenizer.eos_token_id]
                    for s in params.stop:
                        ids = self.tokenizer.encode(s, add_special_tokens=False)
                        if ids:
                            all_stop_ids.extend(ids)
                    gen_kwargs["eos_token_id"] = all_stop_ids
        
        return gen_kwargs
    
    @remote_function(collect='flatten')
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
        extra_model_inputs: Optional[Dict[str, Any]] = None,
    ) -> SampleResponse:
        """Sample completions using transformers generate()."""
        
        # Switch adapter if needed
        if adapter_uri and self.enable_lora:
            await self._load_adapter(adapter_uri)
        
        # Convert params
        gen_kwargs = self._convert_params(sampling_params)
        gen_kwargs["num_return_sequences"] = num_samples
        gen_kwargs["return_dict_in_generate"] = True
        
        if logprobs or include_prompt_logprobs:
            gen_kwargs["output_scores"] = True
        
        # Prepare input
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Repeat for num_samples
        if num_samples > 1:
            input_ids = input_ids.repeat(num_samples, 1)
            attention_mask = attention_mask.repeat(num_samples, 1)
        
        # Build model inputs
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Add extra model inputs for multimodal (pre-processed by template)
        if extra_model_inputs:
            for key, value in extra_model_inputs.items():
                if hasattr(value, 'to'):
                    model_inputs[key] = value.to(device)
                else:
                    model_inputs[key] = value
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                **gen_kwargs,
            )
        
        # Extract generated sequences
        generated_ids = outputs.sequences
        prompt_len = len(prompt_token_ids)
        
        sequences = []
        for i in range(num_samples):
            gen_tokens = generated_ids[i][prompt_len:].tolist()
            
            # Compute logprobs if requested
            seq_logprobs = None
            if logprobs and hasattr(outputs, 'scores') and outputs.scores:
                seq_logprobs = []
                for j, score in enumerate(outputs.scores):
                    if j >= len(gen_tokens):
                        break
                    log_probs = torch.log_softmax(score[i], dim=-1)
                    token_id = gen_tokens[j]
                    seq_logprobs.append(log_probs[token_id].item())
            
            # Determine stop reason
            stop_reason = "length"
            if gen_tokens and gen_tokens[-1] == self.tokenizer.eos_token_id:
                stop_reason = "stop"
            
            sequences.append(SampledSequence(
                stop_reason=stop_reason,
                tokens=gen_tokens,
                logprobs=seq_logprobs,
            ))
        
        # Compute prompt logprobs if requested
        prompt_logprobs_result = None
        topk_prompt_logprobs_result = None
        if include_prompt_logprobs or topk_prompt_logprobs > 0:
            prompt_logprobs_result, topk_prompt_logprobs_result = await self._compute_prompt_logprobs(
                prompt_token_ids,
                topk=topk_prompt_logprobs if topk_prompt_logprobs > 0 else 1,
            )
        
        return SampleResponse(
            sequences=sequences,
            prompt_logprobs=prompt_logprobs_result,
            topk_prompt_logprobs=topk_prompt_logprobs_result if topk_prompt_logprobs > 0 else None,
        )
    
    async def _compute_prompt_logprobs(
        self,
        prompt_token_ids: List[int],
        topk: int = 1,
    ) -> Tuple[List[Optional[float]], List[Optional[List[Tuple[int, float]]]]]:
        """Compute log probabilities for prompt tokens."""
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        prompt_logprobs: List[Optional[float]] = [None]  # First token has no previous context
        topk_logprobs: List[Optional[List[Tuple[int, float]]]] = [None]
        
        for i in range(1, len(prompt_token_ids)):
            token_id = prompt_token_ids[i]
            prev_logprobs = log_probs[i - 1]
            
            # Logprob for the actual token
            prompt_logprobs.append(prev_logprobs[token_id].item())
            
            # Top-k logprobs
            topk_values, topk_indices = prev_logprobs.topk(topk)
            topk_logprobs.append([
                (idx.item(), val.item())
                for idx, val in zip(topk_indices, topk_values)
            ])
        
        return prompt_logprobs, topk_logprobs
    
    @remote_function()
    async def update_weights(
        self,
        weights: Dict[str, torch.Tensor],
        adapter_name: Optional[str] = None,
    ) -> None:
        """Update model weights."""
        if adapter_name is None:
            # Update base model weights
            self.model.load_state_dict(weights, strict=False)
            logger.info(f"Updated {len(weights)} base model weight tensors")
        else:
            # Update LoRA adapter weights
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                adapter_state_dict = {}
                for key, value in weights.items():
                    if adapter_name in key:
                        adapter_state_dict[key] = value
                if adapter_state_dict:
                    self.model.load_state_dict(adapter_state_dict, strict=False)
                    logger.info(f"Updated {len(adapter_state_dict)} adapter weights for {adapter_name}")
    
    @remote_function()
    async def save_weights_for_sampler(
        self,
        user_id: str,
        weights: Dict[str, torch.Tensor],
        peft_config: Dict[str, Any],
    ) -> str:
        """Save weights as a LoRA adapter for sampling."""
        if not self.enable_lora:
            raise RuntimeError("LoRA not enabled. Set enable_lora=True.")
        
        # Save adapter to disk
        adapter_dir = os.path.join(self._lora_weights_dir, user_id)
        os.makedirs(adapter_dir, exist_ok=True)
        
        import safetensors.torch
        safetensors.torch.save_file(weights, os.path.join(adapter_dir, "adapter_model.safetensors"))
        
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(peft_config, f)
        
        # Track adapter
        uri = f"twinkle://{self._model_id}/lora/{user_id}"
        self._adapters[uri] = {
            "user_id": user_id,
            "path": adapter_dir,
            "peft_config": peft_config,
        }
        
        logger.info(f"Saved LoRA adapter for user {user_id}: {len(weights)} tensors")
        return uri
    
    async def _load_adapter(self, adapter_uri: str) -> None:
        """Load a LoRA adapter."""
        if self._current_adapter == adapter_uri:
            return
        
        if adapter_uri not in self._adapters:
            logger.warning(f"Adapter {adapter_uri} not found")
            return
        
        adapter_info = self._adapters[adapter_uri]
        adapter_path = adapter_info["path"]
        
        from peft import PeftModel, PeftConfig
        
        if not isinstance(self.model, PeftModel):
            # First adapter - wrap the model
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                adapter_name=adapter_info["user_id"],
            )
        else:
            # Load additional adapter
            self.model.load_adapter(adapter_path, adapter_name=adapter_info["user_id"])
            self.model.set_adapter(adapter_info["user_id"])
        
        self._current_adapter = adapter_uri
        logger.info(f"Loaded adapter: {adapter_uri}")
    
    async def sleep(self, **kwargs) -> None:
        pass

    async def wake_up(self, **kwargs) -> None:
        pass