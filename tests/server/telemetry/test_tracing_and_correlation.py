# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the business-layer tracing helper, correlation
keys, and ``ResourceMetricsCollector``.

Properties covered:
- # Feature: server-config-observability-refactor, Property 19: Business-layer span lifecycle
- # Feature: server-config-observability-refactor, Property 20: Span exception handling
- # Feature: server-config-observability-refactor, Property 21: Tracing graceful-degradation equivalence
- # Feature: server-config-observability-refactor, Property 22: Correlation attribute attachment
- # Feature: server-config-observability-refactor, Property 23: Correlation prefix invariant
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from unittest import mock

from twinkle.server.telemetry import correlation
from twinkle.server.telemetry.correlation import CORRELATION_KEYS, PREFIX, set_correlation_attrs
from twinkle.server.telemetry.tracing import _NoopSpan, traced_operation

# ---------- correlation-key prefix invariant ----------------------------- #


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


# ---------- Property 22: attachment of present-only values ------------------ #


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
    ))
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


# ---------- Property 21: NoOp degradation equivalence ---------------------- #


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


# ---------- ResourceMetricsCollector wiring ------------------------------ #


def test_resource_metrics_collector_does_not_raise_without_pynvml() -> None:
    """Collector starts cleanly even when pynvml/GPU is absent."""
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
    in that case."""
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
    from opentelemetry import _logs as _otel_logs
    from opentelemetry import metrics, trace
    from opentelemetry.sdk._logs import LoggingHandler
    from opentelemetry.util._once import Once

    from twinkle.server.config.telemetry import TelemetryConfig
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
        provider.init_telemetry(
            TelemetryConfig(
                enabled=True,
                debug=True,  # debug=True → console exporter, no real OTLP needed
                service_name='twinkle-server-test',
            ))
        root_handlers = [h for h in logging.getLogger().handlers if isinstance(h, LoggingHandler)]
        twinkle_handlers = [h for h in logging.getLogger('twinkle').handlers if isinstance(h, LoggingHandler)]
        assert len(root_handlers) == 1, root_handlers
        assert len(twinkle_handlers) == 1, twinkle_handlers
        assert root_handlers[0] is twinkle_handlers[0], ('root and twinkle should share the same handler instance')
        # The OTLP handler carries the transport-stack feedback-loop filter.
        assert any(isinstance(flt, provider._OTLPTransportFilter) for flt in root_handlers[0].filters), \
            'OTLP transport filter not attached to the handler'
    finally:
        provider.shutdown_telemetry()
        # shutdown should detach from both
        assert all(not isinstance(h, LoggingHandler) for name in ('', 'twinkle')
                   for h in logging.getLogger(name).handlers)


# ---------- inbound HTTP trace-context continuity ------------------ #


def test_inbound_traceparent_continues_trace() -> None:
    """The tracing middleware extracts the inbound ``traceparent`` and starts
    its SERVER span within that context, so a Gateway -> Model / Gateway ->
    Sampler hop continues one trace instead of starting a fresh one.

    The active span's ``trace_id`` is captured inside the request handler (it
    runs nested inside the middleware's SERVER span), which makes the assertion
    independent of whichever global tracer provider another test may have
    installed.

    Validates Requirement 15: inbound HTTP trace-context continuity.
    """
    if not _otel_available():
        pytest.skip('OTEL SDK not installed in test env')

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry import trace as otel_trace

    from twinkle.server.telemetry.tracing import create_tracing_middleware

    app = FastAPI()
    app.middleware('http')(create_tracing_middleware('Model'))

    @app.get('/ping')
    async def ping() -> dict:
        ctx = otel_trace.get_current_span().get_span_context()
        return {'trace_id': format(ctx.trace_id, '032x')}

    # traceparent = version(00)-trace_id(32 hex)-span_id(16 hex)-flags(01)
    injected_trace_id = '0af7651916cd43dd8448eb211c80319c'
    parent_span_id = 'b7ad6b7169203331'
    traceparent = f'00-{injected_trace_id}-{parent_span_id}-01'

    client = TestClient(app)
    resp = client.get('/ping', headers={'traceparent': traceparent})
    assert resp.status_code == 200
    recorded = resp.json()['trace_id']
    assert recorded == injected_trace_id, (
        f'span trace_id {recorded} does not match injected {injected_trace_id}; '
        'the middleware started a fresh trace instead of continuing the upstream one')


def test_inbound_without_traceparent_starts_new_trace() -> None:
    """A request with no trace headers still completes without raising and runs
    the handler under a span on a fresh trace (Requirement 15.3)."""
    if not _otel_available():
        pytest.skip('OTEL SDK not installed in test env')

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry import trace as otel_trace

    from twinkle.server.telemetry.tracing import create_tracing_middleware

    app = FastAPI()
    app.middleware('http')(create_tracing_middleware('Gateway'))

    @app.get('/ping')
    async def ping() -> dict:
        ctx = otel_trace.get_current_span().get_span_context()
        return {'trace_id': format(ctx.trace_id, '032x')}

    client = TestClient(app)
    resp = client.get('/ping')
    assert resp.status_code == 200
    # A trace id is assigned; it is not the injected one from the other test.
    assert resp.json()['trace_id'] != '0af7651916cd43dd8448eb211c80319c'
