# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the mock model backend.

Covers:
- Interface conformance: every required method is present and callable
- Forward determinism + shape match input lengths
- Adapter add/remove round-trip
- Remove-absent raises ``KeyError`` and leaves the record intact
- Model backend dispatch / config validation
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.server.exceptions import ConfigError
from twinkle.server.model.app import MODEL_SELECTOR
from twinkle.server.model.backends.mock_model import TwinkleCompatMockModel

_MODEL_BACKENDS = tuple(MODEL_SELECTOR.builders)

# ---------- Interface conformance ----------------------------------------- #

_REQUIRED_METHODS = (
    'tinker_forward_only',
    'tinker_forward_backward',
    'tinker_step',
    'tinker_calculate_metric',
    'tinker_load',
    'forward_only',
    'forward_backward',
    'forward',
    'calculate_loss',
    'backward',
    'step',
    'zero_grad',
    'lr_step',
    'clip_grad_norm',
    'set_loss',
    'set_optimizer',
    'set_lr_scheduler',
    'set_template',
    'set_processor',
    'add_metric',
    'apply_patch',
    'save',
    'load',
    'resume_from_checkpoint',
    'get_state_dict',
    'get_train_configs',
    'add_adapter',
    'add_adapter_to_model',
    'remove_adapter',
    'has_adapter',
    'upload_to_hub',
)


@pytest.mark.parametrize('method_name', _REQUIRED_METHODS)
def test_property_1_required_method_present(method_name: str) -> None:
    m = TwinkleCompatMockModel('mid')
    assert callable(getattr(m, method_name)), method_name


def test_property_1_constructor_does_not_raise() -> None:
    TwinkleCompatMockModel('mid')


# ---------- Forward determinism + shape ----------------------------------- #


@settings(max_examples=100)
@given(
    seq_lens=st.lists(st.integers(min_value=1, max_value=12), min_size=1, max_size=5),
    seed=st.integers(min_value=0, max_value=99),
)
def test_property_2_forward_only_deterministic_and_shaped(seq_lens: list, seed: int) -> None:
    inputs = [{'tokens': list(range(n))} for n in seq_lens]
    a = TwinkleCompatMockModel('mid', seed=seed)
    b = TwinkleCompatMockModel('mid', seed=seed)
    out_a = a.forward_only(inputs=inputs)
    out_b = b.forward_only(inputs=inputs)
    assert out_a == out_b
    assert len(out_a) == len(inputs)
    for record, n in zip(out_a, seq_lens):
        assert len(record['logprobs']) == n
        assert len(record['elementwise_loss']) == n


@settings(max_examples=100)
@given(seq_lens=st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=4))
def test_property_2_tinker_forward_backward_loss_is_finite(seq_lens: list) -> None:
    m = TwinkleCompatMockModel('mid')
    inputs = [{'tokens': list(range(n))} for n in seq_lens]
    result, loss = m.tinker_forward_backward(inputs=inputs, adapter_name='a', loss_fn='cross_entropy')
    assert isinstance(loss, float)
    assert 0.0 <= loss <= 1.0
    assert len(result) == len(inputs)


# ---------- Adapter round-trip -------------------------------------------- #


@settings(max_examples=100)
@given(
    name=st.text(
        min_size=1, max_size=12, alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-')))
def test_property_3_adapter_add_remove_round_trip(name: str) -> None:
    m = TwinkleCompatMockModel('mid')
    assert not m.has_adapter(name)
    m.add_adapter(name, rank=4)
    assert m.has_adapter(name)
    m.remove_adapter(name)
    assert not m.has_adapter(name)


# ---------- Remove-absent raises + preserves record ----------------------- #


@settings(max_examples=100)
@given(name=st.text(min_size=1, max_size=12))
def test_property_4_remove_absent_raises(name: str) -> None:
    m = TwinkleCompatMockModel('mid')
    pre = dict(m._adapters)
    with pytest.raises(KeyError):
        m.remove_adapter(name)
    assert m._adapters == pre


# ---------- Model backend dispatch ---------------------------------------- #


def test_property_10_mock_dispatch_returns_mock_model() -> None:
    m = MODEL_SELECTOR.construct(MODEL_SELECTOR.validate('mock'), {'model_id': 'mid'})
    assert isinstance(m, TwinkleCompatMockModel)


@settings(max_examples=100)
@given(bad=st.text(min_size=1, max_size=10).filter(lambda s: s not in _MODEL_BACKENDS))
def test_property_10_invalid_backend_raises_config_error(bad: str) -> None:
    """Validation runs BEFORE any backend import / instantiation."""
    with pytest.raises(ConfigError) as exc:
        MODEL_SELECTOR.validate(bad)
    assert exc.value.field == 'backend'
    assert exc.value.value == bad
    assert set(exc.value.allowed) == set(_MODEL_BACKENDS)


@pytest.mark.parametrize('value', [None, ''])
def test_property_10_absent_or_empty_backend_raises(value) -> None:
    with pytest.raises(ConfigError) as exc:
        MODEL_SELECTOR.validate(value)
    assert exc.value.field == 'backend'
