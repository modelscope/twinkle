# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from twinkle.data_format import InputFeature

StopReason = Literal['length', 'stop']


@dataclass
class SamplingParams:
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    logprobs: int = None
    prompt_logprobs: int = None
    num_samples: int = 1

    def __post_init__(self):
        if not isinstance(self.temperature, (int, float)):
            raise ValueError(f'temperature must be a number, got {type(self.temperature)}')
        if self.temperature < 0:
            raise ValueError(f'temperature must be >= 0, got {self.temperature}')

        if not isinstance(self.top_p, (int, float)):
            raise ValueError(f'top_p must be a number, got {type(self.top_p)}')
        if not 0 < self.top_p <= 1:
            raise ValueError(f'top_p must be in range (0, 1], got {self.top_p}')

        if not isinstance(self.top_k, int):
            raise ValueError(f'top_k must be an int, got {type(self.top_k)}')
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f'top_k must be -1 or >= 1, got {self.top_k}')

        if self.logprobs is not None:
            if not isinstance(self.logprobs, int):
                raise ValueError(f'logprobs must be an int or None, got {type(self.logprobs)}')
            if self.logprobs < 0:
                raise ValueError(f'logprobs must be >= 0, got {self.logprobs}')

        if self.prompt_logprobs is not None:
            if not isinstance(self.prompt_logprobs, int):
                raise ValueError(f'prompt_logprobs must be an int or None, got {type(self.prompt_logprobs)}')
            if self.prompt_logprobs < 0:
                raise ValueError(f'prompt_logprobs must be >= 0, got {self.prompt_logprobs}')

        if not isinstance(self.num_samples, int):
            raise ValueError(f'num_samples must be an int, got {type(self.num_samples)}')
        if self.num_samples < 1:
            raise ValueError(f'num_samples must be >= 1, got {self.num_samples}')

        if self.max_tokens is not None:
            if not isinstance(self.max_tokens, int):
                raise ValueError(f'max_tokens must be an int or None, got {type(self.max_tokens)}')
            if self.max_tokens < 1:
                raise ValueError(f'max_tokens must be >= 1, got {self.max_tokens}')

        if not isinstance(self.repetition_penalty, (int, float)):
            raise ValueError(f'repetition_penalty must be a number, got {type(self.repetition_penalty)}')
        if self.repetition_penalty <= 0:
            raise ValueError(f'repetition_penalty must be > 0, got {self.repetition_penalty}')

    def to_vllm(self, **kwargs):
        """Convert to vLLM SamplingParams.
        """
        from vllm import SamplingParams as VLLMSamplingParams

        kwargs = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.num_samples,
            **kwargs,
        }

        if self.max_tokens is not None:
            kwargs['max_tokens'] = self.max_tokens

        if self.seed is not None:
            kwargs['seed'] = self.seed

        if self.top_k > 0:
            kwargs['top_k'] = self.top_k

        if self.repetition_penalty != 1.0:
            kwargs['repetition_penalty'] = self.repetition_penalty

        if self.stop:
            if isinstance(self.stop, str):
                kwargs['stop'] = [self.stop]
            elif isinstance(self.stop, (list, tuple)) and self.stop and isinstance(self.stop[0], int):
                kwargs['stop_token_ids'] = list(self.stop)
            else:
                kwargs['stop'] = list(self.stop)

        if self.logprobs is not None:
            kwargs['logprobs'] = self.logprobs

        if self.prompt_logprobs is not None:
            kwargs['prompt_logprobs'] = self.prompt_logprobs

        vllm_params = VLLMSamplingParams(**kwargs)
        if self.num_samples > 1:
            from vllm.sampling_params import RequestOutputKind
            vllm_params.output_kind = RequestOutputKind.FINAL_ONLY
        return vllm_params

    def to_transformers(self, tokenizer=None) -> Dict[str, Any]:
        """Convert to transformers generate() kwargs."""
        import torch

        gen_kwargs = {
            'do_sample': self.temperature > 0,
            'temperature': self.temperature,
            'top_p': self.top_p,
        }

        if self.max_tokens is not None:
            gen_kwargs['max_new_tokens'] = self.max_tokens
        else:
            gen_kwargs['max_new_tokens'] = 2048

        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.top_k > 0:
            gen_kwargs['top_k'] = self.top_k

        if self.repetition_penalty != 1.0:
            gen_kwargs['repetition_penalty'] = self.repetition_penalty

        if tokenizer is not None:
            gen_kwargs['pad_token_id'] = tokenizer.pad_token_id
            gen_kwargs['eos_token_id'] = tokenizer.eos_token_id

            if self.stop:
                if isinstance(self.stop, str):
                    stop_ids = tokenizer.encode(self.stop, add_special_tokens=False)
                    if stop_ids:
                        gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id] + stop_ids
                elif isinstance(self.stop, (list, tuple)):
                    if self.stop and isinstance(self.stop[0], int):
                        gen_kwargs['eos_token_id'] = [tokenizer.eos_token_id] + list(self.stop)
                    else:
                        all_stop_ids = [tokenizer.eos_token_id]
                        for s in self.stop:
                            ids = tokenizer.encode(s, add_special_tokens=False)
                            if ids:
                                all_stop_ids.extend(ids)
                        gen_kwargs['eos_token_id'] = all_stop_ids

        return gen_kwargs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SamplingParams':
        """Create SamplingParams from a dict."""
        if 'max_new_tokens' in d and 'max_tokens' not in d:
            d['max_tokens'] = d.pop('max_new_tokens')

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**filtered)


@dataclass
class SampledSequence:
    """A single sampled sequence with tokens and logprobs."""
    stop_reason: StopReason
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    decoded: str = None
    new_input_feature: InputFeature = None


@dataclass
class SampleResponse:
    """Response from a sampling request."""
    sequences: Sequence[SampledSequence]
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None
