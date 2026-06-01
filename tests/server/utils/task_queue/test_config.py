# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the Pydantic ``TaskQueueConfig`` (R9).

Covers:
- # Feature: server-config-observability-refactor, Property 16: TaskQueueConfig constraint enforcement
- # Feature: server-config-observability-refactor, Property 17: from_dict equivalence
- # Feature: server-config-observability-refactor, Property 18: TaskQueueConfig defaulting
- Unit checks that the call sites in model/sampler/processor apps still construct the config
  through ``TaskQueueConfig.from_dict``.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import ValidationError

from twinkle.server.utils.task_queue.config import TaskQueueConfig


# ---------- defaults snapshot used by Property 18 (R9.8) ------------------- #

DEFAULTS = {
    'rps_limit': 100.0,
    'tps_limit': 16000.0,
    'window_seconds': 1.0,
    'queue_timeout': 300.0,
    'token_cleanup_interval': 60.0,
    'max_input_tokens': 16000,
}


# ---------- Property 16: constraint enforcement (R9.2-9.5, 9.7) ------------ #


_CONSTRAINED_GE0_FLOATS = ['rps_limit', 'tps_limit', 'queue_timeout', 'token_cleanup_interval']


@settings(max_examples=100)
@given(
    field=st.sampled_from(_CONSTRAINED_GE0_FLOATS),
    bad_value=st.floats(max_value=-1e-6, min_value=-1e6, allow_nan=False, allow_infinity=False),
)
def test_property_16_ge0_floats_reject_negative(field: str, bad_value: float) -> None:
    """Non-negative float fields reject any negative input."""
    with pytest.raises(ValidationError) as exc:
        TaskQueueConfig(**{field: bad_value})
    assert any(field in err['loc'] for err in exc.value.errors())


@settings(max_examples=100)
@given(bad_value=st.floats(max_value=0.0, min_value=-1e6, allow_nan=False, allow_infinity=False))
def test_property_16_window_seconds_rejects_zero_and_negative(bad_value: float) -> None:
    """``window_seconds`` must be strictly > 0."""
    with pytest.raises(ValidationError) as exc:
        TaskQueueConfig(window_seconds=bad_value)
    assert any('window_seconds' in err['loc'] for err in exc.value.errors())


@settings(max_examples=100)
@given(bad_value=st.integers(max_value=0, min_value=-1_000_000))
def test_property_16_max_input_tokens_rejects_lt_1(bad_value: int) -> None:
    """``max_input_tokens`` must be an integer ≥ 1."""
    with pytest.raises(ValidationError) as exc:
        TaskQueueConfig(max_input_tokens=bad_value)
    assert any('max_input_tokens' in err['loc'] for err in exc.value.errors())


@settings(max_examples=100)
@given(
    rps=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    tps=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    win=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    qt=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    cleanup=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    mit=st.integers(min_value=1, max_value=10_000_000),
)
def test_property_16_valid_values_accepted(
    rps: float, tps: float, win: float, qt: float, cleanup: float, mit: int
) -> None:
    """Any value satisfying the constraints constructs successfully."""
    cfg = TaskQueueConfig(
        rps_limit=rps,
        tps_limit=tps,
        window_seconds=win,
        queue_timeout=qt,
        token_cleanup_interval=cleanup,
        max_input_tokens=mit,
    )
    assert cfg.rps_limit == rps
    assert cfg.window_seconds == win
    assert cfg.max_input_tokens == mit


# ---------- Property 17: from_dict equivalence (R9.6) ---------------------- #


_INPUT_DICT_STRATEGY = st.fixed_dictionaries(
    {},
    optional={
        'rps_limit': st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        'tps_limit': st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        'window_seconds': st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        'queue_timeout': st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        'token_cleanup_interval': st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        'max_input_tokens': st.integers(min_value=1, max_value=10_000_000),
        'enabled': st.booleans(),
        'execution_timeout': st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        'token_cleanup_multiplier': st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    },
)


@settings(max_examples=100)
@given(payload=_INPUT_DICT_STRATEGY)
def test_property_17_from_dict_equivalence(payload: dict) -> None:
    """``from_dict`` returns the same instance as ``model_validate``."""
    via_factory = TaskQueueConfig.from_dict(payload)
    via_validate = TaskQueueConfig.model_validate(payload)
    assert via_factory.model_dump() == via_validate.model_dump()


# ---------- Property 18: defaulting (R9.8) --------------------------------- #


def test_property_18_from_dict_with_no_argument() -> None:
    """``from_dict()`` with no argument returns the documented defaults."""
    cfg = TaskQueueConfig.from_dict()
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


def test_property_18_from_dict_with_none() -> None:
    cfg = TaskQueueConfig.from_dict(None)
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


def test_property_18_from_dict_with_empty_dict() -> None:
    cfg = TaskQueueConfig.from_dict({})
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


@settings(max_examples=100)
@given(present=st.sets(st.sampled_from(sorted(DEFAULTS.keys())), max_size=len(DEFAULTS)))
def test_property_18_omitted_fields_take_defaults(present: set) -> None:
    """Fields absent from the dict adopt their documented defaults."""
    payload = {f: DEFAULTS[f] for f in present}
    cfg = TaskQueueConfig.from_dict(payload)
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


# ---------- Unit: extra=forbid + call-site usage --------------------------- #


def test_extra_field_rejected() -> None:
    """``extra='forbid'`` rejects unknown keys (defends R8.2 scoped to this model)."""
    with pytest.raises(ValidationError):
        TaskQueueConfig.from_dict({'unknown_field': 1})


def test_call_site_imports_resolve() -> None:
    """``TaskQueueConfig.from_dict`` is what the apps call — keep that import alive.

    We don't instantiate the deployments (those need Ray Serve runtime); we just
    confirm the call sites import the same name and the factory still produces
    valid configs for the dicts they pass.
    """
    from twinkle.server.utils.task_queue import TaskQueueConfig as Exported

    assert Exported is TaskQueueConfig
    # Mimic the queue_config dicts shipped in cookbook/client/server YAMLs.
    cfg = TaskQueueConfig.from_dict({'rps_limit': 100, 'tps_limit': 100000})
    assert cfg.rps_limit == 100.0
    assert cfg.tps_limit == 100000.0
