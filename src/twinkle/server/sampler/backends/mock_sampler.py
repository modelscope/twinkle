# Copyright (c) ModelScope Contributors. All rights reserved.
"""Numpy-only mock sampler backend.

Implements the same surface as :class:`twinkle.sampler.base.Sampler` —
``sample``, ``apply_patch``, ``add_adapter_to_sampler`` — using only numpy.
The class is intentionally **duck-typed** rather than subclassed because
``twinkle.sampler.__init__`` eagerly imports the vLLM engine, which would
pull torch/CUDA on a CPU-only host.

Outputs are deterministic — keyed by ``(model_id, adapter_name, seed,
prompt_index, sample_index)`` — so repeated calls with the same parameters
produce identical token sequences and logprobs.
"""
from __future__ import annotations
import hashlib
from typing import Any, List, Optional

import numpy as np

# These data containers don't pull torch / vllm.
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.utils.logger import get_logger

logger = get_logger()


def _stable_seed(*parts: Any) -> int:
    """Cross-process-stable numpy seed (uint32) derived from string parts.

    Python's built-in ``hash()`` of a tuple containing strings is salted per
    process (PYTHONHASHSEED), which would make identical sample requests on
    different replicas / restarts produce different outputs. Use a stable
    digest instead.
    """
    canonical = '\x1f'.join(str(p) for p in parts).encode('utf-8')
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:4], 'big')


class MockSampler:
    """Deterministic numpy-only sampler.

    Provides the public methods callable from the sampler app and the Tinker /
    Twinkle handlers; ``has_adapter`` is added for convenience and tests.
    """

    def __init__(self, model_id: str, *, seed: int = 0, vocab_size: int = 32, **kwargs: Any) -> None:
        self.model_id = model_id
        self._seed = int(seed)
        self._vocab_size = int(vocab_size)
        self._adapters: dict[str, Any] = {}
        # Surface (rather than silently swallow) extra ctor kwargs: a real
        # backend signature drift then shows up as a visible DEBUG warning in
        # the mock e2e instead of being discarded without trace.
        if kwargs:
            logger.debug('MockSampler ignoring unknown ctor kwargs: %s', sorted(kwargs))
        # Match the Sampler base attributes so duck-typed callers don't surprise.
        self.engine = None
        self.template = None

    # ----- Sampler interface --------------------------------------------- #

    def sample(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
        **kwargs: Any,
    ) -> list[SampleResponse]:
        # The real ``vLLMSampler.sample`` accepts extra keyword arguments
        # (``adapter_path``, ``adapter_uri``, etc.) that the Tinker / Twinkle
        # handlers always forward. Swallow them here so the mock backend
        # stays callable through the same handler call sites without a
        # TypeError.
        max_tokens = self._resolve_max_tokens(sampling_params)
        if max_tokens is None or max_tokens < 1:
            raise ValueError(f'max_tokens must be >= 1, got {max_tokens!r} '
                             '(set sampling_params.max_tokens to a positive integer)')

        normalized = self._normalize_inputs(inputs)
        responses: list[SampleResponse] = []
        for prompt_idx, _ in enumerate(normalized):
            sequences: list[SampledSequence] = []
            for sample_idx in range(num_samples):
                seed = _stable_seed(self.model_id, adapter_name, self._seed, prompt_idx, sample_idx)
                rng = np.random.default_rng(seed)
                tokens = [int(t) for t in rng.integers(low=0, high=max(1, self._vocab_size), size=max_tokens)]
                logprobs_per_token = rng.uniform(-2.0, 0.0, size=max_tokens).astype(float).tolist()
                # twinkle ``SampledSequence.logprobs`` is
                # ``List[List[Tuple[int, float]]]`` (top-k per position) — the
                # mock returns top-1 with the chosen token. The tinker handler
                # flattens this to a single chosen-token logprob.
                logprobs = [[(tok, float(lp))] for tok, lp in zip(tokens, logprobs_per_token)]
                sequences.append(SampledSequence(
                    stop_reason='length',
                    tokens=tokens,
                    logprobs=logprobs,
                ))
            responses.append(SampleResponse(sequences=sequences))
        return responses

    def apply_patch(self, patch_cls: Any, **kwargs: Any) -> None:
        return None

    def set_template(self, template_cls: Any, **kwargs: Any) -> None:
        self.template = template_cls

    def reset_prefix_cache(self) -> None:
        return None

    # ----- Adapter management -------------------------------------------- #

    def add_adapter_to_sampler(self, adapter_name: str, config: Any) -> None:
        self._adapters[adapter_name] = config

    def has_adapter(self, adapter_name: str) -> bool:
        return adapter_name in self._adapters

    # ----- Helpers ------------------------------------------------------- #

    @staticmethod
    def _normalize_inputs(inputs: Any) -> list[Any]:
        if inputs is None:
            return [None]
        if isinstance(inputs, list):
            return inputs if inputs else [None]
        return [inputs]

    @staticmethod
    def _resolve_max_tokens(params: SamplingParams | None) -> int | None:
        if params is None:
            return None
        return getattr(params, 'max_tokens', None)
