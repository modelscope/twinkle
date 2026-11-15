# Copyright (c) ModelScope Contributors. All rights reserved.
"""Archived pipelineTQ surface. Collected by nothing: see ``collect_ignore_glob`` in tests/conftest.py.

``PartialRolloutMixin`` lived in ``twinkle/sampler/partial_rollout.py`` until the streaming RL rewrite
(75e969ab) removed it along with the pipeline that used it, and ``f1d396a5`` restored that pipeline
verbatim under ``async_rl/pipelineTQ/`` as a code archive -- unreferenced and without an
``__init__.py``, so the package never imports it. This test is kept beside the archived code it
exercised rather than deleted, because the archive is the reference for what the retry/resume/merge
loop did; it cannot be imported, which is exactly why it lives under an ignored glob.
"""
import asyncio

from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.partial_rollout import PartialRolloutMixin


def test_partial_rollout_carries_state_across_retries_and_merge():

    class Harness(PartialRolloutMixin):
        template = type('Decoder', (), {'decode': staticmethod(str)})()

    def response(tokens, stop_reason, input_ids, state):
        return SampleResponse(sequences=[SampledSequence(
            tokens=tokens,
            stop_reason=stop_reason,
            logprobs=[-0.1] * len(tokens),
            new_input_feature={'input_ids': input_ids},
            new_state=state,
        )])

    responses = iter([
        response([3], 'abort', [1, 2, 3], 'state-1'),
        response([4], 'stop', [1, 2, 3, 4], 'state-2'),
    ])
    seen_states = []

    async def run_attempt(current_input, current_state, sampling_params, attempt):
        seen_states.append(current_state)
        return next(responses), attempt

    outcome = asyncio.run(Harness()._run_partial_rollout(
        {'input_ids': [1, 2]},
        SamplingParams(max_tokens=2),
        run_attempt=run_attempt,
        original_state='state-0',
        allow_partial_rollout=True,
        max_retries=1,
    ))
    assert seen_states == ['state-0', 'state-1']
    assert outcome.response.sequences[0].tokens == [3, 4]
    assert outcome.response.sequences[0].new_state == 'state-2'
