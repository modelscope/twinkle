# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from twinkle.data_format import InputFeature

StopReason = Literal['length', 'stop', 'abort', 'error']


@dataclass
class SamplingParams:
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    # Whether what ``stop`` matched stays in the output. vLLM drops it by
    # default -- both the string form and the token-id form, since v1's
    # detokenizer excludes the final token whenever a stop terminated the
    # request -- which is wrong for a stop that is part of the syntax being
    # generated. Stopping a tool-using agent at '</tool_call>' so it reads one
    # observation before deciding the next call is exactly that case: without
    # this, every turn the policy is trained on ends on an unclosed
    # '<tool_call>' block.
    include_stop_str_in_output: bool = False
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
            if self.max_tokens < 0:
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
            if self.include_stop_str_in_output:
                kwargs['include_stop_str_in_output'] = True

        if self.logprobs is not None:
            kwargs['logprobs'] = self.logprobs

        if self.prompt_logprobs is not None:
            kwargs['prompt_logprobs'] = self.prompt_logprobs

        vllm_params = VLLMSamplingParams(**kwargs)
        if self.num_samples > 1:
            from vllm.sampling_params import RequestOutputKind
            vllm_params.output_kind = RequestOutputKind.FINAL_ONLY
        return vllm_params

    def to_sglang(self, **kwargs) -> Dict[str, Any]:
        """Convert to the dict sglang takes as its ``sampling_params``.

        sglang calls the token budget ``max_new_tokens``, the seed ``sampling_seed``, and keeps stop
        token ids separate from stop strings. ``logprobs``/``prompt_logprobs`` are deliberately absent:
        sglang requests those per generate() call rather than through the sampling parameters, so
        :class:`~twinkle.sampler.SGLangEngine` translates them instead.
        """
        params = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.num_samples,
            **kwargs,
        }

        if self.max_tokens is not None:
            params['max_new_tokens'] = self.max_tokens

        if self.seed is not None:
            params['sampling_seed'] = self.seed

        if self.top_k > 0:
            params['top_k'] = self.top_k

        if self.repetition_penalty != 1.0:
            params['repetition_penalty'] = self.repetition_penalty

        if self.stop:
            if isinstance(self.stop, str):
                params['stop'] = [self.stop]
            elif isinstance(self.stop, (list, tuple)) and self.stop and isinstance(self.stop[0], int):
                params['stop_token_ids'] = list(self.stop)
            else:
                params['stop'] = list(self.stop)

        return params

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
class SamplingMask:
    """CSR token support sets aligned with sampled sequence tokens."""
    token_ids: List[int]
    offsets: List[int]


@dataclass
class SampledSequence:
    """A single sampled sequence with tokens and logprobs."""
    stop_reason: StopReason
    tokens: List[int]
    logprobs: Optional[List[List[Tuple[int, float]]]] = None
    decoded: str = None
    new_input_feature: InputFeature = None
    routed_experts: Optional[Any] = None
    sampling_mask: Optional[SamplingMask] = None


@dataclass
class SampleResponse:
    """Response from a sampling request."""
    sequences: Sequence[SampledSequence]
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None


#: The pooling heads a backend can serve, in vLLM's vocabulary -- and deliberately only the two vLLM's
#: engine accepts for a whole-sequence forward. ``embed`` returns a sentence vector, ``classify`` the
#: per-class logits/probs. A cross-encoder relevance *score* is NOT a separate task: vLLM serves it as
#: ``classify`` on a scoring model (its own score/rerank API builds ``PoolingParams(task='classify')``),
#: and it is flagged with :attr:`PoolingParams.is_cross_encoder` for the backend (sglang) that routes on
#: the flag rather than inferring it from the model. dev maps its ``task_type`` onto these
#: (embedding->embed, seq_cls/reranker->classify). A *generative* reranker is not a pooling task at all
#: -- it is a decoder-only causal LM scored by the yes/no logprob difference of its first generated
#: token, so it runs on the generation path, never through ``encode``.
PoolingTask = Literal['embed', 'classify']


def pooling_to_list(raw: Any, normalize: bool = False) -> List[float]:
    """Coerce a backend's pooled output to a flat ``List[float]``.

    The three pooling backends disagree on the container: vLLM hands back a torch tensor under
    ``outputs.data``, sglang a python list / numpy array under ``embedding``, and a scalar score arrives
    as a bare number rather than a one-element list. Everything downstream (Ray's result boundary, jsonl
    output) wants plain floats, so the lot is flattened here once instead of at each call site.

    ``normalize`` L2-normalises the vector, which is what an embedding model served for retrieval needs
    and what the backends do not all do by default.
    """
    values = raw
    if hasattr(values, 'detach'):
        values = values.detach()
    if hasattr(values, 'float'):
        try:
            values = values.float()
        except (TypeError, ValueError, RuntimeError):
            pass
    if hasattr(values, 'cpu'):
        try:
            values = values.cpu()
        except (TypeError, ValueError, RuntimeError):
            pass
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if isinstance(values, (int, float)):
        values = [float(values)]
    else:
        values = [float(v) for v in values]
    if normalize:
        import math
        norm = math.sqrt(sum(v * v for v in values))
        if norm > 0:
            values = [v / norm for v in values]
    return values


@dataclass
class PoolingParams:
    """Parameters for a pooling (non-generative) forward -- the counterpart of :class:`SamplingParams`.

    Where ``SamplingParams`` says how to decode tokens, this says which pooled head to run and how to
    post-process it. ``task`` uses vLLM's pooling vocabulary directly so the mapping to a backend is a
    pass-through rather than a second translation table.

    ``use_activation`` applies the head's activation (sigmoid on a cross-encoder score, softmax on
    class logits); the backends spell it differently (vLLM ``use_activation``/older ``activation``), so
    :meth:`to_vllm` picks whichever the installed version takes. ``is_cross_encoder`` marks the request
    as a (query, document) relevance score rather than a plain classify; vLLM infers this from the
    scoring model so the flag is a no-op there, but sglang routes on it explicitly (it wants the raw
    text pair, not token ids). ``dimensions`` truncates an embedding (Matryoshka-style); ``normalize``
    L2-normalises the returned vector.
    """
    task: PoolingTask = 'embed'
    use_activation: bool = False
    is_cross_encoder: bool = False
    dimensions: Optional[int] = None
    normalize: bool = False

    def __post_init__(self):
        valid = ('embed', 'classify')
        if self.task not in valid:
            raise ValueError(f'task must be one of {valid}, got {self.task!r}')
        if self.dimensions is not None:
            if not isinstance(self.dimensions, int) or isinstance(self.dimensions, bool):
                raise ValueError(f'dimensions must be an int or None, got {type(self.dimensions)}')
            if self.dimensions <= 0:
                raise ValueError(f'dimensions must be > 0, got {self.dimensions}')

    def to_vllm(self, **kwargs):
        """Convert to vLLM's ``PoolingParams``, keeping only the fields the installed version takes."""
        import inspect

        from vllm.pooling_params import PoolingParams as VLLMPoolingParams

        params = dict(kwargs)
        signature = inspect.signature(VLLMPoolingParams).parameters
        if 'task' in signature:
            params['task'] = self.task
        if self.use_activation:
            if 'use_activation' in signature:
                params['use_activation'] = True
            elif 'activation' in signature:
                params['activation'] = True
        if self.dimensions is not None and 'dimensions' in signature:
            params['dimensions'] = self.dimensions
        return VLLMPoolingParams(**params)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PoolingParams':
        """Create PoolingParams from a dict, ignoring keys that are not fields."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class PoolingResponse:
    """Response from a pooling (encode) request.

    ``data`` is a flat list of floats whose meaning follows ``task``: an embedding vector for ``embed``,
    or the per-class logits/probs for ``classify`` (a cross-encoder relevance score is the single value
    a ``classify`` scoring model returns). Plain python so it crosses Ray's result boundary and
    serialises to jsonl without further conversion.
    """
    data: List[float]
    task: PoolingTask = 'embed'
    prompt_token_ids: Optional[List[int]] = None
