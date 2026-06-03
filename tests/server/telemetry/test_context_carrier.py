# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trace context carrier round-trip tests."""
from __future__ import annotations

import pytest
from unittest import mock

from twinkle.server.telemetry import context_carrier
from twinkle.server.telemetry.context_carrier import activate_carrier, make_carrier


def _otel_available() -> bool:
    try:
        from opentelemetry import trace  # noqa: F401
    except Exception:
        return False
    return True


# ---------- NoOp path (R13.4 / R18.3) ------------------------------------- #


def test_make_carrier_returns_empty_dict_when_otel_absent() -> None:
    with mock.patch.object(context_carrier, '_OTEL_AVAILABLE', False):
        assert make_carrier() == {}


def test_activate_carrier_with_none_is_safe_noop() -> None:
    with activate_carrier(None):
        pass
    with activate_carrier({}):
        pass


def test_activate_carrier_when_otel_absent_is_noop() -> None:
    with mock.patch.object(context_carrier, '_OTEL_AVAILABLE', False):
        with activate_carrier({'traceparent': 'whatever'}):
            pass


# ---------- Property 24: round-trip (R13.1, R13.2) ------------------------ #


def test_property_24_carrier_round_trip(in_memory_span_exporter) -> None:
    """Active context → make_carrier → activate_carrier → child span shares
    the same trace id (R13.1, R13.2)."""
    from opentelemetry import trace

    in_memory_span_exporter.clear()
    tracer = trace.get_tracer('twinkle.test')

    carrier: dict[str, object] = {}
    parent_trace_id: int | None = None
    with tracer.start_as_current_span('caller') as parent:
        parent_trace_id = parent.get_span_context().trace_id
        carrier = make_carrier()

    # Sanity — the carrier carries something OTEL recognizes (traceparent).
    assert carrier and any(k.lower() == 'traceparent' for k in carrier.keys())

    # On the receiving side, activating the carrier and starting a span
    # should yield a span whose trace id equals the parent's.
    with activate_carrier(carrier):
        with tracer.start_as_current_span('callee') as child:
            child_trace_id = child.get_span_context().trace_id

    assert parent_trace_id == child_trace_id


def test_property_24_empty_carrier_starts_fresh_trace(in_memory_span_exporter) -> None:
    """An empty / None carrier means: start a new trace."""
    from opentelemetry import trace

    in_memory_span_exporter.clear()
    tracer = trace.get_tracer('twinkle.test')

    with activate_carrier(None):
        with tracer.start_as_current_span('orphan') as span:
            tid = span.get_span_context().trace_id
            assert tid != 0
