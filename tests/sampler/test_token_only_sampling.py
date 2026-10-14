"""Tinker consumes encoded tokens, independently of native template settings."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.sampler.vllm_sampler import vLLMSampler


class _OtherClientTemplate:
    def get_vllm_input_ids(self, *args):
        raise AssertionError('Token-only sampling rewrote encoded input IDs')

    def concat_input_feature(self, *args):
        raise AssertionError('Token-only sampling built unused training features')

    def decode(self, *args):
        raise AssertionError('Token-only sampling decoded unused text')


@pytest.mark.asyncio
@pytest.mark.parametrize('template', [None, _OtherClientTemplate()])
@pytest.mark.parametrize('logprobs_only', [False, True])
async def test_token_only_sampling_needs_no_template(template, logprobs_only):
    sampler = object.__new__(vLLMSampler)
    sampler.template = template
    sampler.engine = SimpleNamespace(sample=AsyncMock(return_value=SampleResponse(
        prompt_token_ids=[7],
        sequences=[SampledSequence(tokens=[8], logprobs=[[(8, -.1)]], stop_reason='length')],
        prompt_logprobs=[-.2],
    )))
    result = await sampler._sample_single(
        {'input_ids': [7]}, SamplingParams(max_tokens=1), token_only=True, logprobs_only=logprobs_only)
    assert sampler.engine.sample.call_args.kwargs['prompt'] == [7]
    assert result.prompt_token_ids == [7] and result.prompt_logprobs == [-.2]
    assert result.sequences[0].tokens == ([] if logprobs_only else [8])
    assert result.sequences[0].logprobs == (None if logprobs_only else [[(8, -.1)]])
    assert result.sequences[0].decoded is None and result.sequences[0].new_input_feature is None
