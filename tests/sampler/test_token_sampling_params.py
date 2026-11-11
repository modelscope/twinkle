"""Preserve sampled-token log-probs and text-dependent stop semantics."""
import pytest

from twinkle.data_format.sampling import SamplingParams
from twinkle.sampler.vllm_sampler.vllm_engine import _token_sampling_params


@pytest.mark.parametrize('requested,expected', [(None, None), (0, 0), (1, 0), (3, 3)])
def test_sampled_only_does_not_request_an_unused_top1(requested, expected):
    source = SamplingParams(logprobs=requested)
    params = _token_sampling_params(source)
    assert params.logprobs == expected
    assert source.logprobs == requested
    assert params.detokenize is False


@pytest.mark.parametrize('stop', ['END', ['END'], [42]])
def test_text_stop_keeps_detokenization_but_token_stop_does_not_need_it(stop):
    params = _token_sampling_params(SamplingParams(stop=stop))
    assert params.detokenize is (stop != [42])


def test_bad_words_keep_detokenization():
    assert _token_sampling_params(SamplingParams(), bad_words=['word']).detokenize is True
