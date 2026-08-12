# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the mock sampler.

Covers:
- Interface conformance (required methods are present and callable)
- Output length + logprob count match ``max_tokens``
- Determinism: identical inputs produce identical outputs across calls
- ``max_tokens < 1`` and missing ``sampling_params`` rejected
- Adapter record updates persist
- Dispatch / config validation rejects unknown sampler types
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.data_format import InputFeature, SamplingParams
from twinkle.server.exceptions import ConfigError
from twinkle.server.sampler.app import SAMPLER_SELECTOR
from twinkle.server.sampler.backends.mock_sampler import MockSampler

_SAMPLER_TYPES = tuple(SAMPLER_SELECTOR.builders)

# ---------- Interface conformance ----------------------------------------- #

_REQUIRED_METHODS = ('sample', 'apply_patch', 'add_adapter_to_sampler', 'has_adapter', 'set_template',
                     'reset_prefix_cache')


@pytest.mark.parametrize('method', _REQUIRED_METHODS)
def test_required_method_present(method: str) -> None:
    s = MockSampler('mid')
    assert callable(getattr(s, method))


# ---------- Output length + logprob count --------------------------------- #


@settings(max_examples=100)
@given(
    max_tokens=st.integers(min_value=1, max_value=20),
    num_samples=st.integers(min_value=1, max_value=4),
)
def test_output_length_and_logprob_count(max_tokens: int, num_samples: int) -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1, 2, 3])
    responses = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name='a', num_samples=num_samples)
    assert len(responses) == 1
    seqs = responses[0].sequences
    assert len(seqs) == num_samples
    for seq in seqs:
        assert len(seq.tokens) == max_tokens
        assert len(seq.logprobs) == max_tokens


# ---------- Determinism --------------------------------------------------- #


@settings(max_examples=100)
@given(
    max_tokens=st.integers(min_value=1, max_value=10),
    num_samples=st.integers(min_value=1, max_value=3),
    adapter=st.sampled_from(['', 'a', 'lora-1']),
)
def test_determinism(max_tokens: int, num_samples: int, adapter: str) -> None:
    s = MockSampler('mid', seed=42)
    inp = InputFeature(input_ids=[1, 2, 3])
    r1 = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name=adapter, num_samples=num_samples)
    r2 = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name=adapter, num_samples=num_samples)
    assert r1 == r2


# ---------- Invalid max_tokens rejected ----------------------------------- #


@settings(max_examples=50)
@given(bad=st.integers(max_value=0, min_value=-1000))
def test_max_tokens_lt_1_raises(bad: int) -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    with pytest.raises(ValueError) as exc:
        s.sample(inp, SamplingParams(max_tokens=bad))
    assert 'max_tokens' in str(exc.value)


def test_no_sampling_params_raises() -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    with pytest.raises(ValueError):
        s.sample(inp, sampling_params=None)


# ---------- Adapter record update ----------------------------------------- #


@settings(max_examples=100)
@given(
    name=st.text(
        min_size=1, max_size=12, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-')))
def test_add_adapter_to_sampler(name: str) -> None:
    s = MockSampler('mid')
    assert not s.has_adapter(name)
    s.add_adapter_to_sampler(name, {'rank': 4})
    assert s.has_adapter(name)
    assert s._adapters[name] == {'rank': 4}


# ---------- Sampler backend dispatch -------------------------------------- #


def test_mock_dispatch_returns_mock_sampler() -> None:
    s = SAMPLER_SELECTOR.construct(SAMPLER_SELECTOR.validate('mock'), {'model_id': 'mid'})
    assert isinstance(s, MockSampler)


@settings(max_examples=100)
@given(bad=st.text(min_size=1, max_size=10).filter(lambda s: s not in _SAMPLER_TYPES))
def test_invalid_sampler_type_raises_config_error(bad: str) -> None:
    """Validation runs BEFORE any sampler import / instantiation."""
    with pytest.raises(ConfigError) as exc:
        SAMPLER_SELECTOR.validate(bad)
    assert exc.value.field == 'sampler_type'
    assert exc.value.value == bad
    assert set(exc.value.allowed) == set(_SAMPLER_TYPES)


@pytest.mark.parametrize('value', [None, ''])
def test_absent_or_empty_sampler_type_raises(value) -> None:
    with pytest.raises(ConfigError) as exc:
        SAMPLER_SELECTOR.validate(value)
    assert exc.value.field == 'sampler_type'


# ---------- Streaming interface -------------------------------------------- #


def test_sample_stream_method_present() -> None:
    s = MockSampler('mid')
    assert callable(getattr(s, 'sample_stream'))


@settings(max_examples=100, deadline=None)
@given(max_tokens=st.integers(min_value=1, max_value=20))
def test_sample_stream_yields_correct_count(max_tokens: int) -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1, 2, 3])
    chunks = list(s.sample_stream(inp, SamplingParams(max_tokens=max_tokens)))
    assert len(chunks) == max_tokens
    for delta_text, finish_reason in chunks[:-1]:
        assert isinstance(delta_text, str)
        assert finish_reason is None
    last_text, last_reason = chunks[-1]
    assert isinstance(last_text, str)
    assert last_reason is not None


def test_sample_stream_rejects_bad_max_tokens() -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    with pytest.raises(ValueError):
        list(s.sample_stream(inp, SamplingParams(max_tokens=0)))


# ---------- Multi-turn contract knobs -------------------------- #


def test_default_new_input_feature_appends_sampled_tokens() -> None:
    """Default behaviour now carries a non-empty new_input_feature that appends
    the round's sampled tokens onto the input pif's input_ids."""
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1, 2, 3])
    seq = s.sample(inp, SamplingParams(max_tokens=4))[0].sequences[0]
    assert seq.new_input_feature is not None
    assert 'input_ids' in seq.new_input_feature
    assert seq.new_input_feature['input_ids'] == [1, 2, 3] + list(seq.tokens)
    # Prior context stays non-trainable; sampled tokens are trainable (their ids).
    assert seq.new_input_feature['labels'] == [-100, -100, -100] + list(seq.tokens)
    assert seq.new_input_feature['length'] == 3 + len(seq.tokens)


def test_default_backward_compatible_stop_reason_and_decoded() -> None:
    """With no knobs configured, stop_reason stays 'length' and decoded None."""
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    seq = s.sample(inp, SamplingParams(max_tokens=2))[0].sequences[0]
    assert seq.stop_reason == 'length'
    assert seq.decoded is None


def test_configurable_stop_reason_via_ctor() -> None:
    s = MockSampler('mid', stop_reason='stop')
    inp = InputFeature(input_ids=[1])
    seq = s.sample(inp, SamplingParams(max_tokens=2))[0].sequences[0]
    assert seq.stop_reason == 'stop'


def test_stop_reason_per_call_kwarg_overrides_ctor() -> None:
    s = MockSampler('mid', stop_reason='stop')
    inp = InputFeature(input_ids=[1])
    seq = s.sample(inp, SamplingParams(max_tokens=2), stop_reason='length')[0].sequences[0]
    assert seq.stop_reason == 'length'


def test_tool_call_text_injected_only_on_configured_turns() -> None:
    """tool_call_text is emitted as decoded on turn 1 only (default), driving a
    'sample -> tool -> next round -> terminate' loop."""
    s = MockSampler('mid', stop_reason='stop', tool_call_text='<tool_call>x</tool_call>')
    inp = InputFeature(input_ids=[1, 2])
    # Round 1: tool call injected.
    seq1 = s.sample(inp, SamplingParams(max_tokens=2))[0].sequences[0]
    assert seq1.decoded == '<tool_call>x</tool_call>'
    # Round 2: no injection -> decoded None (loop can terminate on empty parse).
    seq2 = s.sample(inp, SamplingParams(max_tokens=2))[0].sequences[0]
    assert seq2.decoded is None


def test_tool_call_turns_multiple_rounds() -> None:
    s = MockSampler('mid', stop_reason='stop', tool_call_text='TC', tool_call_turns=(1, 3))
    inp = InputFeature(input_ids=[1])
    decoded = [s.sample(inp, SamplingParams(max_tokens=1))[0].sequences[0].decoded for _ in range(3)]
    assert decoded == ['TC', None, 'TC']
