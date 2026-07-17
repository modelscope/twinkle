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
from typing import Any, Iterable

from twinkle import remote_class, remote_function
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.utils.logger import get_logger
from twinkle.utils.seed import stable_seed

logger = get_logger()


@remote_class()
class MockSampler:
    """Deterministic mock sampler for CPU-only testing.

    Beyond the base sampler surface, this backend exposes a few opt-in knobs so
    a local (CPU-only) multi-turn rollout e2e can be driven without a GPU:

    - ``stop_reason`` (default ``'length'``): the ``SampledSequence.stop_reason``
      emitted for every sampled sequence. ``'length'`` terminates a multi-turn
      loop immediately; ``'stop'`` lets the loop proceed to tool-call parsing.
    - ``tool_call_text`` (default ``None``): when set, this text is emitted as
      ``SampledSequence.decoded`` on the configured turns so the rollout's
      ``template.parse_tool_call`` can produce a tool call and exercise the
      "sample -> tool -> bridge -> sample" control flow.
    - ``tool_call_turns`` (default ``(1,)`` when ``tool_call_text`` is set):
      the 1-based turn indices on which ``tool_call_text`` is injected. A turn
      corresponds to one ``sample()`` invocation (one multi-turn round).

    All three knobs may be supplied at construction time or overridden per call
    via ``sample()`` keyword arguments or matching attributes on
    ``sampling_params``. When none are configured the backend keeps its previous
    behaviour exactly (``stop_reason='length'``, ``decoded=None``), so existing
    callers are unaffected. Outputs stay deterministic and CPU-only.
    """

    def __init__(
        self,
        model_id: str,
        *,
        seed: int = 0,
        vocab_size: int = 32,
        stop_reason: str = 'length',
        tool_call_text: str | None = None,
        tool_call_turns: Iterable[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self._seed = int(seed)
        self._vocab_size = int(vocab_size)
        self._adapters: dict[str, Any] = {}
        # Multi-turn control-flow knobs (see class docstring).
        self._stop_reason = str(stop_reason)
        self._tool_call_text = tool_call_text
        self._tool_call_turns = self._normalize_turns(tool_call_turns, tool_call_text)
        # Per-round counter: incremented once per ``sample()`` call so that
        # ``tool_call_turns`` can address individual multi-turn rounds. Only
        # consulted when tool-call injection is active, so the default path
        # stays fully stateless and deterministic.
        self._round = 0
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

        # Resolve per-call multi-turn knobs (precedence: explicit sample()
        # kwargs > sampling_params attributes > ctor defaults) and advance the
        # round counter that ``tool_call_turns`` addresses.
        stop_reason = self._resolve_stop_reason(sampling_params, kwargs)
        tool_call_text = self._resolve_tool_call_text(sampling_params, kwargs)
        tool_call_turns = self._resolve_tool_call_turns(sampling_params, kwargs, tool_call_text)
        self._round += 1
        inject_tool_call = tool_call_text is not None and self._round in tool_call_turns
        decoded = tool_call_text if inject_tool_call else None

        normalized = self._normalize_inputs(inputs)
        responses: list[SampleResponse] = []
        for prompt_idx, pif in enumerate(normalized):
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
                sequences.append(
                    SampledSequence(
                        stop_reason=stop_reason,
                        tokens=tokens,
                        logprobs=logprobs,
                        decoded=decoded,
                        new_input_feature=self._build_new_input_feature(pif, tokens),
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

    @staticmethod
    def _normalize_turns(turns: Iterable[int] | None, tool_call_text: str | None) -> frozenset[int]:
        """Normalise the ``tool_call_turns`` config into a ``frozenset[int]``.

        When ``tool_call_text`` is set but no turns are given, default to
        ``{1}`` — inject the tool call exactly once, on the first round.
        """
        if turns is None:
            return frozenset({1}) if tool_call_text is not None else frozenset()
        return frozenset(int(t) for t in turns)

    def _resolve_stop_reason(self, params: SamplingParams | None, kwargs: dict[str, Any]) -> str:
        if 'stop_reason' in kwargs and kwargs['stop_reason'] is not None:
            return str(kwargs['stop_reason'])
        override = getattr(params, 'stop_reason', None)
        if override is not None:
            return str(override)
        return self._stop_reason

    def _resolve_tool_call_text(self, params: SamplingParams | None, kwargs: dict[str, Any]) -> str | None:
        if 'tool_call_text' in kwargs:
            return kwargs['tool_call_text']
        override = getattr(params, 'tool_call_text', None)
        if override is not None:
            return override
        return self._tool_call_text

    def _resolve_tool_call_turns(
        self,
        params: SamplingParams | None,
        kwargs: dict[str, Any],
        tool_call_text: str | None,
    ) -> frozenset[int]:
        if 'tool_call_turns' in kwargs and kwargs['tool_call_turns'] is not None:
            return self._normalize_turns(kwargs['tool_call_turns'], tool_call_text)
        override = getattr(params, 'tool_call_turns', None)
        if override is not None:
            return self._normalize_turns(override, tool_call_text)
        # Fall back to the ctor-configured turns, but keep the "default to {1}
        # when text is injected via a per-call override" behaviour consistent.
        if tool_call_text is not None and not self._tool_call_turns:
            return frozenset({1})
        return self._tool_call_turns

    @staticmethod
    def _build_new_input_feature(pif: Any, tokens: list[int]) -> dict[str, Any]:
        """Deterministically append the freshly sampled ``tokens`` to ``pif``.

        Produces a plain-dict ``InputFeature`` that carries the running context
        for the next multi-turn round: ``input_ids`` is the prior prompt plus
        this round's sampled tokens, and ``labels`` marks the sampled tokens as
        trainable (their own ids) while prior/context positions stay ``-100``.
        This mirrors the shape a real sampler's ``concat_input_feature`` yields,
        which the multi-turn rollout relies on (it reads
        ``new_input_feature.input_ids`` and counts trainable ``labels``).

        The mock does not depend on a template, so the append is a simple,
        deterministic list concatenation performed entirely on the CPU.
        """
        feat: dict[str, Any] = dict(pif) if isinstance(pif, dict) else {}
        prev_ids = list(feat.get('input_ids') or [])
        prev_labels = feat.get('labels')
        if prev_labels is not None and len(prev_labels) == len(prev_ids):
            labels = list(prev_labels)
        else:
            # No (or misaligned) prior labels: treat the entire prior context as
            # non-trainable so only this round's sampled tokens count.
            labels = [-100] * len(prev_ids)

        feat['input_ids'] = prev_ids + list(tokens)
        feat['labels'] = labels + list(tokens)
        feat['length'] = len(feat['input_ids'])
        return feat
