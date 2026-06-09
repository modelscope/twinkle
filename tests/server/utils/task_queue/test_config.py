# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the Pydantic ``TaskQueueConfig``.

Pins constraint enforcement, default-value defaulting, and ``extra='forbid'``
behavior. The class is constructed directly from validated YAML by
``ApplicationSpec`` (typed end-to-end after Task 27), so there is no
``from_dict`` revival path to exercise.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from twinkle.server.utils.task_queue.config import TaskQueueConfig

# ---------- defaults snapshot used by the default-value test -------------- #

DEFAULTS = {
    'rps_limit': 100.0,
    'tps_limit': 16000.0,
    'window_seconds': 1.0,
    'queue_timeout': 300.0,
    'token_cleanup_interval': 60.0,
    'max_input_tokens': 16000,
}

# ---------- Property 16: constraint enforcement ----------------------------- #

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
def test_property_16_valid_values_accepted(rps: float, tps: float, win: float, qt: float, cleanup: float,
                                           mit: int) -> None:
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


# ---------- defaulting --------------------------------------------------- #


def test_default_construction_uses_documented_defaults() -> None:
    """``TaskQueueConfig()`` with no args returns the documented defaults."""
    cfg = TaskQueueConfig()
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


@settings(max_examples=100)
@given(present=st.sets(st.sampled_from(sorted(DEFAULTS.keys())), max_size=len(DEFAULTS)))
def test_omitted_fields_take_defaults(present: set) -> None:
    """Fields absent from the kwargs adopt their documented defaults."""
    payload = {f: DEFAULTS[f] for f in present}
    cfg = TaskQueueConfig(**payload)
    for field, value in DEFAULTS.items():
        assert getattr(cfg, field) == value, field


# ---------- Unit: extra=forbid -------------------------------------------- #


def test_extra_field_rejected() -> None:
    """``extra='forbid'`` rejects unknown keys."""
    with pytest.raises(ValidationError):
        TaskQueueConfig(unknown_field=1)
