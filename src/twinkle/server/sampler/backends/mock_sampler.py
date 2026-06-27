# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mock sampler backend.

Implements the same surface as :class:`twinkle.sampler.base.Sampler` —
``sample``, ``apply_patch``, ``add_adapter_to_sampler`` — using only numpy.
The class is duck-typed rather than subclassed to keep the module
self-contained.

Outputs are deterministic — keyed by ``(model_id, adapter_name, seed,
prompt_index, sample_index)`` — so repeated calls with the same parameters
produce identical token sequences and logprobs.
"""
from __future__ import annotations

import numpy as np
import time
from typing import Any

from twinkle import remote_class, remote_function
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.utils.logger import get_logger
from twinkle.utils.seed import stable_seed

logger = get_logger()


@remote_class()
class MockSampler:
    """Deterministic mock sampler for CPU-only testing."""

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

    @remote_function()
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
                seed = stable_seed(self.model_id, adapter_name, self._seed, prompt_idx, sample_idx)
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

    @remote_function()
    def sample_stream(
        self,
        inputs: Any,
        sampling_params: SamplingParams | None = None,
        adapter_name: str = '',
        adapter_path: str | None = None,
    ):
        """Yield (delta_text, finish_reason) tuples one token at a time."""
        max_tokens = self._resolve_max_tokens(sampling_params)
        if max_tokens is None or max_tokens < 1:
            raise ValueError(f'max_tokens must be >= 1, got {max_tokens!r}')

        seed = stable_seed(self.model_id, adapter_name, 0, 0)
        rng = np.random.default_rng(seed)
        tokens = [int(t) for t in rng.integers(low=0, high=max(1, self._vocab_size), size=max_tokens)]

        for i, tok in enumerate(tokens):
            is_last = i == len(tokens) - 1
            time.sleep(0.05)
            yield str(tok), ('length' if is_last else None)

    def sample_stream_to_queue(self, queue, inputs, sampling_params=None, adapter_name='', adapter_path=None):
        """Push streaming deltas to a cross-process Ray queue."""
        from . import stream_to_queue
        stream_to_queue(self, queue, inputs, sampling_params, adapter_name, adapter_path)

    @remote_function()
    def apply_patch(self, patch_cls: Any, **kwargs: Any) -> None:
        return None

    @remote_function()
    def set_template(self, template_cls: Any, **kwargs: Any) -> None:
        self.template = template_cls

    @remote_function()
    def reset_prefix_cache(self) -> None:
        return None

    # ----- Adapter management -------------------------------------------- #

    @remote_function()
    def add_adapter_to_sampler(self, adapter_name: str, config: Any) -> None:
        self._adapters[adapter_name] = config

    @remote_function()
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
