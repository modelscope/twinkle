# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the business-layer tracing helper, correlation
keys, and ``ResourceMetricsCollector`` (R10, R11, R12, R18).

Properties covered:
- # Feature: server-config-observability-refactor, Property 19: Business-layer span lifecycle
- # Feature: server-config-observability-refactor, Property 20: Span exception handling
- # Feature: server-config-observability-refactor, Property 21: Tracing graceful-degradation equivalence
- # Feature: server-config-observability-refactor, Property 22: Correlation attribute attachment
- # Feature: server-config-observability-refactor, Property 23: Correlation prefix invariant
"""
from __future__ import annotations

from unittest import mock

import pytest
from hypothesis import given, settings, strategies as st

from twinkle.server.telemetry import correlation
from twinkle.server.telemetry.correlation import (
    CORRELATION_KEYS,
    PREFIX,
    set_correlation_attrs,
)
from twinkle.server.telemetry.tracing import _NoopSpan, traced_operation


# ---------- Property 23: prefix invariant (R11.3) ------------------------- #


@pytest.mark.parametrize('key', CORRELATION_KEYS)
def test_property_23_prefix_invariant(key: str) -> None:
    assert key.startswith(PREFIX), key


def test_property_23_helper_constants_complete() -> None:
    expected = {
        'twinkle.session_id',
        'twinkle.model_id',
        'twinkle.replica_id',
        'twinkle.token_id',
        'twinkle.sampling_session_id',
        'twinkle.base_model',
    }
    assert set(CORRELATION_KEYS) == expected


# ---------- Property 22: attachment of present-only values (R11.1, R11.2) - #


class _RecordingSpan:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attrs[key] = value


@settings(max_examples=100)
@given(
    payload=st.fixed_dictionaries(
        {},
        optional={
            correlation.SESSION_ID: st.one_of(st.none(), st.text(min_size=1, max_size=8)),
            correlation.MODEL_ID: st.one_of(st.none(), st.text(min_size=1, max_size=8)),
            correlation.REPLICA_ID: st.one_of(st.none(), st.text(min_size=1, max_size=8)),
            correlation.TOKEN_ID: st.one_of(st.none(), st.text(min_size=1, max_size=8)),
        },
    )
)
def test_property_22_set_correlation_attrs_skips_none(payload: dict) -> None:
    span = _RecordingSpan()
    set_correlation_attrs(span, payload)
    expected = {k: v for k, v in payload.items() if v is not None}
    assert span.attrs == expected


def test_property_22_noop_span_safe() -> None:
    """``set_correlation_attrs`` is a no-op on a NoOp span (no SDK installed)."""
    span = _NoopSpan()
    set_correlation_attrs(span, {correlation.SESSION_ID: 's1'})
    # NoOp span has no recording surface — passing None / empty mapping is also safe.
    set_correlation_attrs(None, {correlation.SESSION_ID: 's1'})
    set_correlation_attrs(span, None)


# ---------- Property 21: NoOp degradation equivalence (R10.5, R18.3) ------ #


def test_property_21_noop_yields_same_result_as_active() -> None:
    """When OTEL is absent, ``traced_operation`` runs the body and returns the
    body's result identically to when OTEL is active."""
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', False):
        with traced_operation('op') as span:
            assert isinstance(span, _NoopSpan)
            result = sum(range(5))
        assert result == 10


def test_property_21_noop_propagates_exceptions() -> None:
    """NoOp path still re-raises the original exception unchanged."""
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', False):
        with pytest.raises(RuntimeError, match='boom'):
            with traced_operation('op'):
                raise RuntimeError('boom')


# ---------- Property 19/20: span lifecycle + exception handling ----------- #


def _otel_available() -> bool:
    try:
        from opentelemetry import trace as _otel_trace  # noqa: F401
    except Exception:
        return False
    return True


def test_property_19_span_lifecycle(in_memory_span_exporter) -> None:
    """When OTEL is present, a span is started before and ended after the block."""
    in_memory_span_exporter.clear()
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', True):
        with traced_operation('op.under.test', attrs={correlation.SESSION_ID: 's1'}):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    matches = [s for s in spans if s.name == 'op.under.test']
    assert matches
    assert matches[-1].attributes.get(correlation.SESSION_ID) == 's1'


def test_property_20_exception_recorded_and_reraised(in_memory_span_exporter) -> None:
    """Exception inside the block is recorded on the span and re-raised."""
    in_memory_span_exporter.clear()
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', True):
        with pytest.raises(ValueError, match='boom'):
            with traced_operation('op.exc'):
                raise ValueError('boom')

    spans = [s for s in in_memory_span_exporter.get_finished_spans() if s.name == 'op.exc']
    assert spans, 'span was not exported'
    span = spans[-1]
    assert span.status.status_code.name == 'ERROR'
    assert any('exception' in evt.name.lower() for evt in span.events)


# ---------- ResourceMetricsCollector wiring (R12.1, R12.2, R12.3, R18.4) -- #


def test_resource_metrics_collector_does_not_raise_without_pynvml() -> None:
    """Collector starts cleanly even when pynvml/GPU is absent (R12.3)."""
    from twinkle.server.telemetry import resource_metrics

    with mock.patch.object(resource_metrics, '_PYNVML_AVAILABLE', False):
        collector = resource_metrics.ResourceMetricsCollector()
        collector.maybe_start()  # must not raise
        # No GPU gauges registered when pynvml is absent.
        assert all(not g.startswith('twinkle.gpu.') for g in collector.registered_gauges)


def test_resource_metrics_collector_registers_named_gauges_when_psutil_present() -> None:
    from twinkle.server.telemetry import resource_metrics

    if not resource_metrics._PSUTIL_AVAILABLE:
        pytest.skip('psutil not installed in test env')
    collector = resource_metrics.ResourceMetricsCollector()
    collector.maybe_start()
    # System CPU + system memory + process memory always present when psutil is.
    expected = {
        'twinkle.system.cpu.utilization',
        'twinkle.system.memory.usage_bytes',
        'twinkle.process.memory.usage_bytes',
    }
    assert expected.issubset(set(collector.registered_gauges))


def test_resource_metrics_collector_idempotent() -> None:
    from twinkle.server.telemetry import resource_metrics

    collector = resource_metrics.ResourceMetricsCollector()
    collector.maybe_start()
    pre = list(collector.registered_gauges)
    collector.maybe_start()
    assert collector.registered_gauges == pre


def test_worker_init_starts_collector() -> None:
    """``ensure_telemetry_initialized`` calls the resource collector even when
    telemetry is disabled — the collector silently records to a NoOp meter
    in that case (R12.2)."""
    from twinkle.server.telemetry import resource_metrics, worker_init

    # Force the worker_init guard to re-run and clear the global collector
    # so we observe a fresh ``maybe_start`` call.
    worker_init._worker_initialized = False
    resource_metrics.reset_collector_for_tests()

    sentinel = mock.MagicMock()
    sentinel.maybe_start = mock.MagicMock()

    with mock.patch.object(resource_metrics, 'get_collector', return_value=sentinel) as get_spy:
        worker_init.ensure_telemetry_initialized()

    assert get_spy.call_count >= 1
    assert sentinel.maybe_start.call_count == 1


def test_init_telemetry_attaches_handler_to_twinkle_logger() -> None:
    """``init_telemetry`` must attach the OTLP ``LoggingHandler`` to BOTH
    the root logger AND the ``twinkle`` logger.

    The ``twinkle.utils.logger`` module configures the ``twinkle`` namespace
    with ``propagate=False`` and its own StreamHandler — so an OTLP handler
    bound only to root would never see any ``twinkle.*`` log records, and
    the entire server's logs would be invisible in Loki / OTLP backends.
    """
    import logging
    from opentelemetry import _logs as _otel_logs, metrics, trace
    from opentelemetry.sdk._logs import LoggingHandler
    from opentelemetry.util._once import Once

    from twinkle.server.telemetry import provider

    # Reset all OTel global guards so init_telemetry runs cleanly.
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None
    metrics._METER_PROVIDER_SET_ONCE = Once()
    metrics._METER_PROVIDER = None
    if hasattr(_otel_logs, '_LOGGER_PROVIDER_SET_ONCE'):
        _otel_logs._LOGGER_PROVIDER_SET_ONCE = Once()
        _otel_logs._LOGGER_PROVIDER = None
    provider._initialized = False

    # Clear stale handlers that might be attached from prior tests.
    for name in ('', 'twinkle'):
        for h in list(logging.getLogger(name).handlers):
            if isinstance(h, LoggingHandler):
                logging.getLogger(name).removeHandler(h)

    try:
        provider.init_telemetry(provider.TelemetryConfig(
            enabled=True, debug=True,  # debug=True → console exporter, no real OTLP needed
            service_name='twinkle-server-test',
        ))
        root_handlers = [
            h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler)
        ]
        twinkle_handlers = [
            h for h in logging.getLogger('twinkle').handlers if isinstance(h, LoggingHandler)
        ]
        assert len(root_handlers) == 1, root_handlers
        assert len(twinkle_handlers) == 1, twinkle_handlers
        assert root_handlers[0] is twinkle_handlers[0], (
            'root and twinkle should share the same handler instance'
        )
    finally:
        provider.shutdown_telemetry()
        # shutdown should detach from both
        assert all(
            not isinstance(h, LoggingHandler)
            for name in ('', 'twinkle')
            for h in logging.getLogger(name).handlers
        )


def test_pyproject_declares_telemetry_extras() -> None:
    """``pyproject.toml`` declares ``psutil`` and ``pynvml`` as telemetry extras (R12.4)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    text = (repo_root / 'pyproject.toml').read_text()
    assert 'telemetry =' in text
    assert 'psutil' in text
    assert 'pynvml' in text


def test_grafana_dashboard_includes_resource_panels() -> None:
    """Grafana dashboard JSON ships CPU / Memory / GPU panels (R12.5)."""
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    dashboard = json.loads(
        (repo_root / 'cookbook' / 'observability' / 'grafana' / 'dashboards'
         / 'twinkle-overview.json').read_text()
    )
    titles = ' | '.join(p['title'].lower() for p in dashboard['panels'])
    for required in ('cpu', 'memory', 'gpu utilization', 'gpu memory'):
        assert required in titles, f'dashboard missing panel containing {required!r}'

    # Each resource gauge name must be referenced by at least one panel target.
    targets = ' | '.join(
        t.get('expr', '')
        for p in dashboard['panels']
        for t in p.get('targets', [])
    )
    for metric in (
        'twinkle_system_cpu_utilization',
        'twinkle_system_memory_usage_bytes',
        'twinkle_process_memory_usage_bytes',
        'twinkle_gpu_utilization',
        'twinkle_gpu_memory_usage_bytes',
    ):
        assert metric in targets, f'dashboard does not query metric {metric!r}'
