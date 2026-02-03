# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union, Dict, Any


StopReason = Literal["length", "stop"]


@dataclass
class SamplingParams:
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    
    def to_vllm(self, *, num_samples: int = 1, logprobs: bool = True, prompt_logprobs: int = 0):
        """Convert to vLLM SamplingParams.
        
        Args:
            num_samples: Number of completions per prompt (vLLM's 'n' parameter).
            logprobs: Whether to return logprobs for generated tokens.
            prompt_logprobs: Number of prompt token logprobs to return.
        """
        from vllm import SamplingParams as VLLMSamplingParams
        
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": num_samples,
        }

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        
        if self.seed is not None:
            kwargs["seed"] = self.seed
        
        if self.top_k > 0:
            kwargs["top_k"] = self.top_k
        
        if self.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.repetition_penalty
        
        if self.stop:
            if isinstance(self.stop, str):
                kwargs["stop"] = [self.stop]
            elif isinstance(self.stop, (list, tuple)) and self.stop and isinstance(self.stop[0], int):
                kwargs["stop_token_ids"] = list(self.stop)
            else:
                kwargs["stop"] = list(self.stop)
        
        if logprobs:
            kwargs["logprobs"] = 0
        
        if prompt_logprobs > 0:
            kwargs["prompt_logprobs"] = prompt_logprobs
        
        vllm_params = VLLMSamplingParams(**kwargs)
        if num_samples > 1:
            from vllm.sampling_params import RequestOutputKind
            vllm_params.output_kind = RequestOutputKind.FINAL_ONLY
        return vllm_params
    
    def to_transformers(self, tokenizer=None) -> Dict[str, Any]:
        """Convert to transformers generate() kwargs."""
        import torch
        
        gen_kwargs = {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        if self.max_tokens is not None:
            gen_kwargs["max_new_tokens"] = self.max_tokens
        else:
            gen_kwargs["max_new_tokens"] = 2048

        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        if self.top_k > 0:
            gen_kwargs["top_k"] = self.top_k
        
        if self.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty
        
        if tokenizer is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            
            if self.stop:
                if isinstance(self.stop, str):
                    stop_ids = tokenizer.encode(self.stop, add_special_tokens=False)
                    if stop_ids:
                        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + stop_ids
                elif isinstance(self.stop, (list, tuple)):
                    if self.stop and isinstance(self.stop[0], int):
                        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(self.stop)
                    else:
                        all_stop_ids = [tokenizer.eos_token_id]
                        for s in self.stop:
                            ids = tokenizer.encode(s, add_special_tokens=False)
                            if ids:
                                all_stop_ids.extend(ids)
                        gen_kwargs["eos_token_id"] = all_stop_ids
        
        return gen_kwargs
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SamplingParams":
        """Create SamplingParams from a dict."""
        if "max_new_tokens" in d and "max_tokens" not in d:
            d["max_tokens"] = d.pop("max_new_tokens")

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        
        return cls(**filtered)


@dataclass
class SampledSequence:
    """A single sampled sequence with tokens and logprobs."""
    stop_reason: StopReason
    tokens: List[int]
    logprobs: Optional[List[float]] = None


@dataclass
class SampleResponse:
    """Response from a sampling request."""
    sequences: Sequence[SampledSequence]
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None
