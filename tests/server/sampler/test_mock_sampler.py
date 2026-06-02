# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the numpy-only mock sampler (R2, R3, R4).

Properties covered:
- # Feature: server-config-observability-refactor, Property 5: Mock sampler interface conformance
- # Feature: server-config-observability-refactor, Property 6: Mock sampler output length and logprob count
- # Feature: server-config-observability-refactor, Property 7: Mock sampler determinism
- # Feature: server-config-observability-refactor, Property 8: Mock sampler rejects invalid max tokens
- # Feature: server-config-observability-refactor, Property 9: Mock sampler adapter record update
- # Feature: server-config-observability-refactor, Property 11: Sampler backend dispatch
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pathlib import Path

from twinkle.data_format import InputFeature, SamplingParams
from twinkle.server.exceptions import ConfigError
from twinkle.server.sampler.app import _SAMPLER_TYPES, _dispatch_sampler_backend, _validate_sampler_type
from twinkle.server.sampler.backends.mock_sampler import MockSampler

# ---------- Property 5: interface conformance (R2.1) ---------------------- #

_REQUIRED_METHODS = ('sample', 'apply_patch', 'add_adapter_to_sampler', 'has_adapter')


@pytest.mark.parametrize('method', _REQUIRED_METHODS)
def test_property_5_required_method_present(method: str) -> None:
    s = MockSampler('mid')
    assert callable(getattr(s, method))


# ---------- Property 6: output length + logprob count (R2.3, R2.4) -------- #


@settings(max_examples=100)
@given(
    max_tokens=st.integers(min_value=1, max_value=20),
    num_samples=st.integers(min_value=1, max_value=4),
)
def test_property_6_output_length_and_logprob_count(max_tokens: int, num_samples: int) -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1, 2, 3])
    responses = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name='a', num_samples=num_samples)
    assert len(responses) == 1
    seqs = responses[0].sequences
    assert len(seqs) == num_samples
    for seq in seqs:
        assert len(seq.tokens) == max_tokens
        assert len(seq.logprobs) == max_tokens


# ---------- Property 7: determinism (R2.5, R4.5) -------------------------- #


@settings(max_examples=100)
@given(
    max_tokens=st.integers(min_value=1, max_value=10),
    num_samples=st.integers(min_value=1, max_value=3),
    adapter=st.sampled_from(['', 'a', 'lora-1']),
)
def test_property_7_determinism(max_tokens: int, num_samples: int, adapter: str) -> None:
    s = MockSampler('mid', seed=42)
    inp = InputFeature(input_ids=[1, 2, 3])
    r1 = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name=adapter, num_samples=num_samples)
    r2 = s.sample(inp, SamplingParams(max_tokens=max_tokens), adapter_name=adapter, num_samples=num_samples)
    assert r1 == r2


# ---------- Property 8: invalid max_tokens rejected (R2.6) ---------------- #


@settings(max_examples=50)
@given(bad=st.integers(max_value=0, min_value=-1000))
def test_property_8_max_tokens_lt_1_raises(bad: int) -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    with pytest.raises(ValueError) as exc:
        s.sample(inp, SamplingParams(max_tokens=bad))
    assert 'max_tokens' in str(exc.value)


def test_property_8_no_sampling_params_raises() -> None:
    s = MockSampler('mid')
    inp = InputFeature(input_ids=[1])
    with pytest.raises(ValueError):
        s.sample(inp, sampling_params=None)


# ---------- Property 9: adapter record update (R2.7) ---------------------- #


@settings(max_examples=100)
@given(
    name=st.text(
        min_size=1, max_size=12, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-')))
def test_property_9_add_adapter_to_sampler(name: str) -> None:
    s = MockSampler('mid')
    assert not s.has_adapter(name)
    s.add_adapter_to_sampler(name, {'rank': 4})
    assert s.has_adapter(name)
    assert s._adapters[name] == {'rank': 4}


# ---------- Property 11: Sampler dispatch (R3.4-3.6, R3.10) --------------- #


def test_property_11_mock_dispatch_returns_mock_sampler() -> None:
    s = _dispatch_sampler_backend(_validate_sampler_type('mock'), {'model_id': 'mid'})
    assert isinstance(s, MockSampler)


@settings(max_examples=100)
@given(bad=st.text(min_size=1, max_size=10).filter(lambda s: s not in _SAMPLER_TYPES))
def test_property_11_invalid_sampler_type_raises_config_error(bad: str) -> None:
    """Validation runs BEFORE any sampler import / instantiation (R3.10)."""
    with pytest.raises(ConfigError) as exc:
        _validate_sampler_type(bad)
    assert exc.value.field == 'sampler_type'
    assert exc.value.value == bad
    assert set(exc.value.allowed) == set(_SAMPLER_TYPES)


@pytest.mark.parametrize('value', [None, ''])
def test_property_11_absent_or_empty_sampler_type_raises(value) -> None:
    with pytest.raises(ConfigError) as exc:
        _validate_sampler_type(value)
    assert exc.value.field == 'sampler_type'


# ---------- No direct vllm import (R2.2) ---------------------------------- #


def test_mock_sampler_module_does_not_directly_import_vllm() -> None:
    """Static check: ``mock_sampler.py`` must not import ``vllm`` directly."""
    src = Path(
        __file__).resolve().parents[3] / 'src' / 'twinkle' / 'server' / 'sampler' / 'backends' / 'mock_sampler.py'
    text = src.read_text()
    for forbidden in ('import vllm', 'from vllm'):
        assert forbidden not in text, f'mock_sampler.py contains {forbidden!r}'


