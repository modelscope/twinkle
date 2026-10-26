# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the PartialRolloutMixin resume loop (driven by a fake run_attempt)."""
import asyncio
from types import SimpleNamespace

import pytest

from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.partial_rollout import PartialRolloutMixin


class _Harness(PartialRolloutMixin):

    def __init__(self):
        self.template = SimpleNamespace(decode=lambda tokens: ''.join(chr(t) for t in tokens))


def _response(stop_reason, tokens, *, new_input_feature=None, new_state=None, logprobs=None):
    return SampleResponse(
        prompt_token_ids=[99],
        sequences=[SampledSequence(
            stop_reason=stop_reason,
            tokens=tokens,
            logprobs=logprobs,
            decoded='',
            new_input_feature=new_input_feature,
            new_state=new_state,
        )],
    )


def _run(harness, run_attempt, **kwargs):
    params = SamplingParams(max_tokens=kwargs.pop('max_tokens', 8))
    return asyncio.run(harness._run_partial_rollout(
        {'input_ids': [99]}, params, run_attempt=run_attempt, **kwargs))


class TestSingleAttempt:

    def test_clean_stop_returns_immediately(self):
        harness = _Harness()
        calls = []

        async def run_attempt(current_input, current_state, params, attempt):
            calls.append((current_input, attempt))
            return _response('stop', [1, 2]), 'state-0'

        outcome = _run(harness, run_attempt)
        assert outcome.attempts == 1
        assert outcome.was_aborted is False
        assert outcome.resumed_partial_output is False
        assert outcome.attempt_states == ('state-0',)
        assert outcome.response.sequences[0].tokens == [1, 2]
        assert calls == [({'input_ids': [99]}, 0)]

    def test_error_without_partial_rollout_exhausts_retries_and_raises(self):
        harness = _Harness()
        calls = []

        async def run_attempt(current_input, current_state, params, attempt):
            calls.append(attempt)
            return _response('error', []), None

        with pytest.raises(RuntimeError, match='generation failed after 2 attempts'):
            _run(harness, run_attempt, max_retries=1)
        assert calls == [0, 1]


class TestResumeAcrossAborts:

    def test_abort_with_tokens_resumes_from_the_new_feature(self):
        harness = _Harness()
        calls = []

        async def run_attempt(current_input, current_state, params, attempt):
            calls.append((current_input, current_state, params.max_tokens))
            if attempt == 0:
                return _response('abort', [1, 2], new_input_feature={'input_ids': [99, 1, 2]},
                                 new_state='state-a'), 'attempt-state-0'
            return _response('stop', [3]), 'attempt-state-1'

        outcome = _run(harness, run_attempt, allow_partial_rollout=True, max_retries=2)
        # The second attempt continues from the first segment's feature and state,
        # with the token budget reduced by what the first segment already produced.
        assert calls[1] == ({'input_ids': [99, 1, 2]}, 'state-a', 8 - 2)
        assert outcome.attempts == 2
        assert outcome.was_aborted is True
        assert outcome.resumed_partial_output is True
        assert outcome.attempt_states == ('attempt-state-0', 'attempt-state-1')
        # Segments are merged in attempt order.
        assert outcome.response.sequences[0].tokens == [1, 2, 3]
        assert outcome.response.sequences[0].stop_reason == 'stop'
        assert outcome.response.prompt_token_ids == [99]

    def test_abort_without_tokens_resets_to_the_original_input(self):
        harness = _Harness()
        calls = []

        async def run_attempt(current_input, current_state, params, attempt):
            calls.append(current_input)
            if attempt == 0:
                return _response('abort', []), None
            return _response('stop', [5]), 'final'

        outcome = _run(harness, run_attempt, allow_partial_rollout=True, max_retries=2)
        assert calls == [{'input_ids': [99]}, {'input_ids': [99]}]
        assert outcome.attempts == 2
        assert outcome.resumed_partial_output is False
        assert outcome.response.sequences[0].tokens == [5]

    def test_reaching_the_token_budget_reports_length(self):
        harness = _Harness()

        async def run_attempt(current_input, current_state, params, attempt):
            if attempt == 0:
                return _response('abort', [1, 2, 3], new_input_feature={'continued': True}), None
            return _response('abort', [4, 5, 6], new_input_feature={'continued': True}), None

        outcome = _run(harness, run_attempt, allow_partial_rollout=True, max_retries=5, max_tokens=6)
        # The six generated tokens equal the budget, so the merged response reports
        # 'length' rather than the last segment's 'abort'.
        assert outcome.response.sequences[0].tokens == [1, 2, 3, 4, 5, 6]
        assert outcome.response.sequences[0].stop_reason == 'length'

    def test_handover_abort_does_not_consume_the_retry_budget(self):
        harness = _Harness()
        harness._ensure_pause_state()
        harness._generation_paused = True

        async def run_attempt(current_input, current_state, params, attempt):
            if attempt == 0:
                return _response('abort', [1], new_input_feature={'continued': True}), None
            return _response('stop', [2]), 'final'

        # max_retries=0 would fail any ordinary retry; a resource handover continues.
        outcome = _run(harness, run_attempt, allow_partial_rollout=True, max_retries=0)
        assert outcome.attempts == 2
        assert outcome.response.sequences[0].tokens == [1, 2]

    def test_merge_joins_logprobs_in_attempt_order(self):
        harness = _Harness()

        async def run_attempt(current_input, current_state, params, attempt):
            if attempt == 0:
                return _response('abort', [1], logprobs=[[(1, -0.5)]],
                                 new_input_feature={'continued': True}), None
            return _response('stop', [2], logprobs=[[(2, -1.0)]]), None

        outcome = _run(harness, run_attempt, allow_partial_rollout=True, max_retries=2)
        assert outcome.response.sequences[0].logprobs == [[(1, -0.5)], [(2, -1.0)]]
        # The merged decode covers the full token span.
        assert outcome.response.sequences[0].decoded == chr(1) + chr(2)


class TestExceptionFromAttempt:

    def test_raising_attempt_is_retried_and_chained_into_the_final_error(self):
        harness = _Harness()
        calls = []

        async def run_attempt(current_input, current_state, params, attempt):
            calls.append(attempt)
            if attempt == 0:
                raise ValueError('engine exploded')
            return _response('stop', [7]), 'ok'

        outcome = _run(harness, run_attempt, max_retries=1)
        assert calls == [0, 1]
        assert outcome.attempts == 2
        assert outcome.response.sequences[0].tokens == [7]
