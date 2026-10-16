# Copyright (c) ModelScope Contributors. All rights reserved.
"""Engine-agnostic partial rollout (interrupt/resume) for the core samplers.

Deep-buffer async RL republishes the policy weights while generations are still in flight. With in-place
weight sync the sampler holds a single live weight copy, so an in-flight generation would otherwise decode
its second half under half-updated weights -- its logprobs then match no consistent policy version and
importance-sampling correction cannot repair them. Partial rollout fixes this: on republish the driver
aborts every in-flight request (:meth:`PartialRolloutMixin.abort_all_inflight`), each aborted generation
comes back with ``stop_reason='abort'`` plus the tokens it produced so far, and the resume loop below
continues it from ``new_input_feature`` on the fresh weights, merging the segments at the end.

The abort/resume *data contract* is already engine-agnostic -- ``SampledSequence`` carries ``stop_reason``
and ``new_input_feature``, and both the vLLM and SGLang engines map an aborted request to
``stop_reason='abort'`` -- so this mixin holds the one shared resume/merge loop plus the in-flight request
registry, and each backend supplies only two engine hooks: ``_sample_single`` (already present) and an
``async def abort_request(request_id)`` on its engine wrapper. It lives here in the core ``twinkle.sampler``
package so the plain :class:`~twinkle.sampler.vllm_sampler.vLLMSampler` /
:class:`~twinkle.sampler.sglang_sampler.SGLangSampler` and the higher-level ``twinkle_agentic``
TransferQueue samplers share one implementation, so a resume-logic fix never has to be mirrored across
backends or layers.

Contract for a concrete subclass: be decorated with ``@remote_class()``, own a running background event
loop at ``self._async_loop``, initialise ``self._inflight_request_ids`` to a set, expose ``self.engine``
with an ``async def abort_request(request_id)``, and implement ``async def _sample_single(...)``.
"""
from __future__ import annotations

import asyncio
from copy import copy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from twinkle import get_logger, remote_function
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams

logger = get_logger()


@dataclass
class PartialRolloutOutcome:
    """What one sample's resume loop produced.

    ``attempt_states`` carries the opaque per-attempt state the caller's ``run_attempt`` returned -- the
    TransferQueue path uses it for the ``RolloutPolicy`` of each segment so it can attribute merged tokens
    to policy versions; the plain core path leaves it ``None``. This mixin never inspects it, so a caller
    can thread whatever per-attempt context it needs through the resume loop without the loop knowing what
    it is.
    """
    response: SampleResponse
    attempt_states: tuple[Any, ...]
    attempts: int
    was_aborted: bool
    resumed_partial_output: bool


class PartialRolloutMixin:
    """The shared interrupt/resume loop and in-flight abort registry, engine-agnostic.

    See the module docstring for the concrete-subclass contract. Everything except the single
    ``_sample_single`` coroutine and the engine's ``abort_request`` is the same regardless of the engine,
    so it is defined once here.
    """

    #: Default resume budget for the core submission path (``_generate_inputs``): how many times an aborted
    #: generation may be re-submitted from its resume point, and the pause before each retry. The
    #: TransferQueue samplers override these per-instance from their pipeline runtime config; the plain
    #: core samplers inherit these defaults. Sized for a deep buffer that republishes a couple of times
    #: over one generation's lifetime (each republish aborts it once, so the budget bounds the staleness a
    #: single generation can survive and still finish).
    rollout_max_retries: int = 2
    rollout_retry_delay_s: float = 0.5

    # ------------------------------------------------------------------
    # In-flight request registry (the abort-on-publish trigger surface)
    # ------------------------------------------------------------------
    def _register_inflight(self, request_id: str) -> None:
        """Record a request id as in-flight so :meth:`abort_all_inflight` can interrupt it.

        Called from ``_sample_single`` on the sampler's background event loop; the registry is only ever
        touched from that single loop thread, so a plain set needs no locking.
        """
        ids = getattr(self, '_inflight_request_ids', None)
        if ids is None:
            ids = set()
            self._inflight_request_ids = ids
        ids.add(request_id)

    def _unregister_inflight(self, request_id: str) -> None:
        ids = getattr(self, '_inflight_request_ids', None)
        if ids is not None:
            ids.discard(request_id)

    async def _abort_request(self, request_id: str) -> None:
        """Abort one in-flight engine request; it then finishes with ``stop_reason='abort'`` + partials."""
        await self.engine.abort_request(request_id)

    async def _abort_all_inflight(self) -> int:
        """Abort every in-flight request on this worker; returns how many abort calls were issued.

        Snapshots the registry first: a request may finish on its own between the snapshot and its abort,
        which the engine treats as a no-op, so a per-request failure is logged and skipped rather than
        aborting the whole sweep.
        """
        request_ids = list(getattr(self, '_inflight_request_ids', ()) or ())
        aborted = 0
        for request_id in request_ids:
            try:
                await self._abort_request(request_id)
                aborted += 1
            except Exception as error:
                logger.warning('partial rollout: failed to abort in-flight request %s: %s', request_id, error)
        return aborted

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def abort_all_inflight(self) -> dict[str, int]:
        """Abort every in-flight generation on this DP worker (called on an in-place weight republish).

        The abort coroutines must run on the engine's own loop, so this schedules ``_abort_all_inflight``
        there and blocks the actor thread on it -- the same loop/thread split the rest of the submission
        API uses, and blocking here is what makes the republish wait until no generation is mid-decode
        before the weights are overwritten.
        """
        future = asyncio.run_coroutine_threadsafe(self._abort_all_inflight(), self._async_loop)
        return {'aborted': future.result()}

    # ------------------------------------------------------------------
    # Resume / merge kernel
    # ------------------------------------------------------------------
    def _merge_partial_responses(
        self,
        responses: list[SampleResponse],
        *,
        stop_reason: Optional[str] = None,
    ) -> SampleResponse:
        """Concatenate the segments of one resumed generation into a single ``SampleResponse``.

        Tokens and per-token logprobs are joined in attempt order; the final segment's ``new_input_feature``
        (prompt + all generated tokens, re-run through the template) and ``routed_experts`` are carried so
        downstream training sees one contiguous sample. ``stop_reason`` overrides the last segment's reason
        -- used to report ``'length'`` when a resume hit the token budget mid-abort.
        """
        sequences = [response.sequences[0] for response in responses]
        tokens = [token for sequence in sequences for token in sequence.tokens]
        logprobs = [logprob for sequence in sequences for logprob in (sequence.logprobs or [])]
        final_sequence = sequences[-1]
        return SampleResponse(
            prompt_token_ids=responses[0].prompt_token_ids,
            sequences=[
                SampledSequence(
                    stop_reason=stop_reason or final_sequence.stop_reason,
                    tokens=tokens,
                    logprobs=logprobs,
                    decoded=self.template.decode(tokens),
                    new_input_feature=final_sequence.new_input_feature,
                    routed_experts=final_sequence.routed_experts,
                )
            ],
        )

    async def _run_partial_rollout(
        self,
        original_input: dict[str, Any],
        sampling_params: SamplingParams,
        *,
        run_attempt: Callable[[Any, SamplingParams, int], Awaitable[tuple[SampleResponse, Any]]],
        allow_partial_rollout: bool = False,
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> PartialRolloutOutcome:
        """Drive one sample's generate-until-finished loop, resuming across aborts when enabled.

        ``run_attempt(current_input, attempt_params, attempt_index)`` performs a single generation attempt
        and returns ``(response, attempt_state)``; it owns everything attempt-specific -- the engine call
        and, on the TransferQueue path, the policy acquire/release around it -- so this loop stays free of
        any control-plane or engine knowledge. On each attempt the token budget is reduced by what earlier
        segments already produced. A response whose ``stop_reason`` is ``'abort'``/``'error'`` is not final:
        with partial rollout on and tokens in hand, the loop records the segment, advances ``current_input``
        to ``new_input_feature`` and retries; otherwise it resets to the original input and retries. A clean
        stop ends the loop, merging any accumulated segments first.
        """
        current_input = original_input
        partial_responses: list[SampleResponse] = []
        partial_states: list[Any] = []
        generated_tokens = 0
        last_error: Optional[Exception] = None
        was_aborted = False
        resumed_partial_output = False

        for attempt in range(max_retries + 1):
            attempt_params = copy(sampling_params)
            if allow_partial_rollout and attempt_params.max_tokens is not None:
                attempt_params.max_tokens -= generated_tokens
            try:
                response, attempt_state = await run_attempt(current_input, attempt_params, attempt)
                sequence = response.sequences[0]
            except Exception as exc:
                last_error = exc
            else:
                if sequence.stop_reason not in {'abort', 'error'}:
                    if not allow_partial_rollout or not partial_responses:
                        return PartialRolloutOutcome(response, (attempt_state, ), attempt + 1, was_aborted,
                                                     resumed_partial_output)
                    partial_responses.append(response)
                    partial_states.append(attempt_state)
                    return PartialRolloutOutcome(
                        self._merge_partial_responses(partial_responses), tuple(partial_states), attempt + 1,
                        was_aborted, resumed_partial_output)

                last_error = RuntimeError(f'generation stopped with {sequence.stop_reason}')
                was_aborted = was_aborted or sequence.stop_reason == 'abort'
                if allow_partial_rollout and sequence.tokens:
                    resumed_partial_output = True
                    partial_responses.append(response)
                    partial_states.append(attempt_state)
                    generated_tokens += len(sequence.tokens)
                    current_input = sequence.new_input_feature
                    if sampling_params.max_tokens is not None and generated_tokens >= sampling_params.max_tokens:
                        return PartialRolloutOutcome(
                            self._merge_partial_responses(partial_responses, stop_reason='length'),
                            tuple(partial_states),
                            attempt + 1,
                            was_aborted,
                            resumed_partial_output,
                        )
                elif not allow_partial_rollout:
                    current_input = original_input

            if attempt < max_retries:
                await asyncio.sleep(retry_delay_s)

        error_detail = f'{type(last_error).__name__}: {last_error}'
        error = RuntimeError(f'generation failed after {max_retries + 1} attempts; last error: {error_detail}')
        raise error from last_error
