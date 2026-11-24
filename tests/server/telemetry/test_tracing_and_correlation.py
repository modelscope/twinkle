# Copyright (c) ModelScope Contributors. All rights reserved.
"""Property + unit tests for the business-layer tracing helper, correlation
keys, and ``ResourceMetricsCollector``.

Properties covered:
- Business-layer span lifecycle
- Span exception handling
- Tracing graceful-degradation equivalence
- Correlation attribute attachment
- Correlation prefix invariant
"""
from __future__ import annotations
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from twinkle.server.config.telemetry import TelemetryConfig
from twinkle.server.telemetry import correlation
from twinkle.server.telemetry.correlation import set_correlation_attrs
from twinkle.server.telemetry.tracing import _NoopSpan, traced_operation

# ---------- attachment of present-only values ------------------------------ #


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
            correlation.TENANT_FINGERPRINT: st.one_of(st.none(), st.text(min_size=1, max_size=8)),
        },
    ))
def test_set_correlation_attrs_skips_none(payload: dict) -> None:
    span = _RecordingSpan()
    set_correlation_attrs(span, payload)
    expected = {k: v for k, v in payload.items() if v is not None}
    assert span.attrs == expected


def test_noop_span_safe() -> None:
    """``set_correlation_attrs`` is a no-op on a NoOp span (no SDK installed)."""
    span = _NoopSpan()
    set_correlation_attrs(span, {correlation.SESSION_ID: 's1'})
    # NoOp span has no recording surface — passing None / empty mapping is also safe.
    set_correlation_attrs(None, {correlation.SESSION_ID: 's1'})
    set_correlation_attrs(span, None)


# ---------- NoOp degradation equivalence ----------------------------------- #


def test_noop_yields_same_result_as_active() -> None:
    """When OTEL is absent, ``traced_operation`` runs the body and returns the
    body's result identically to when OTEL is active."""
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', False):
        with traced_operation('op') as span:
            assert isinstance(span, _NoopSpan)
            result = sum(range(5))
        assert result == 10


def test_noop_propagates_exceptions() -> None:
    """NoOp path still re-raises the original exception unchanged."""
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', False):
        with pytest.raises(RuntimeError, match='boom'):
            with traced_operation('op'):
                raise RuntimeError('boom')


# ---------- span lifecycle + exception handling ---------------------------- #


def _otel_available() -> bool:
    try:
        from opentelemetry import trace as _otel_trace  # noqa: F401
    except Exception:
        return False
    return True


def test_span_lifecycle(in_memory_span_exporter) -> None:
    """When OTEL is present, a span is started before and ended after the block."""
    in_memory_span_exporter.clear()
    with mock.patch('twinkle.server.telemetry.tracing._OTEL_AVAILABLE', True):
        with traced_operation('op.under.test', attrs={correlation.SESSION_ID: 's1'}):
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    matches = [s for s in spans if s.name == 'op.under.test']
    assert matches
    assert matches[-1].attributes.get(correlation.SESSION_ID) == 's1'


def test_exception_recorded_and_reraised(in_memory_span_exporter) -> None:
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
    event, = span.events
    assert event.name == 'exception'
    assert dict(event.attributes) == {'exception.type': 'ValueError'}
    assert not span.status.description


# ---------- ResourceMetricsCollector wiring ------------------------------ #


@pytest.fixture
def resource_meter(monkeypatch):
    from twinkle.server.telemetry import resource_metrics

    meter = mock.Mock()
    monkeypatch.setattr(resource_metrics, 'get_meter', lambda _: meter)
    return meter


def test_resource_metrics_collector_does_not_raise_without_pynvml(resource_meter) -> None:
    from twinkle.server.telemetry import resource_metrics

    with mock.patch.object(resource_metrics, '_PYNVML_AVAILABLE', False):
        collector = resource_metrics.ResourceMetricsCollector(TelemetryConfig())
        collector.maybe_start()
        names = [call.args[0] for call in resource_meter.create_observable_gauge.call_args_list]
        assert all(not name.startswith('twinkle.gpu.') for name in names)


def test_resource_metrics_collector_registers_named_gauges_when_psutil_present(resource_meter) -> None:
    from twinkle.server.telemetry import resource_metrics

    if not resource_metrics._PSUTIL_AVAILABLE:
        pytest.skip('psutil not installed in test env')
    collector = resource_metrics.ResourceMetricsCollector(TelemetryConfig())
    collector.maybe_start()
    expected = {
        'twinkle.system.cpu.utilization',
        'twinkle.system.memory.usage_bytes',
        'twinkle.process.memory.usage_bytes',
    }
    assert expected <= {c.args[0] for c in resource_meter.create_observable_gauge.call_args_list}


def test_resource_metrics_collector_idempotent(resource_meter) -> None:
    from twinkle.server.telemetry import resource_metrics

    collector = resource_metrics.ResourceMetricsCollector(TelemetryConfig())
    collector.maybe_start()
    resource_meter.reset_mock()
    collector.maybe_start()
    resource_meter.create_observable_gauge.assert_not_called()


def test_worker_init_registers_resources_only_with_telemetry(monkeypatch) -> None:
    from twinkle.server.telemetry import resource_metrics, worker_init

    monkeypatch.setattr(worker_init, '_node_metrics', None)
    monkeypatch.delenv('TWINKLE_TELEMETRY_ENABLED', raising=False)
    with (mock.patch.object(resource_metrics, 'ResourceMetricsCollector') as collector,
          mock.patch.object(worker_init, 'init_telemetry', side_effect=(False, False, True)) as init,
          mock.patch.object(worker_init, 'shutdown_telemetry') as shutdown):
        worker_init.ensure_telemetry_initialized()
        collector.assert_not_called()

        config = TelemetryConfig(enabled=True, debug=True, service_name='test-worker',
                                 otlp_endpoint='http://collector:4317', export_interval_ms=250,
                                 resource_attributes={'deployment.environment.name': 'test'})
        for key, value in config.to_env_vars().items():
            monkeypatch.setenv(key, value)
        worker_init.ensure_telemetry_initialized()
        collector.assert_not_called()
        worker_init.ensure_telemetry_initialized()
        worker_init.ensure_telemetry_initialized()
        collector.assert_called_once_with(config)
        collector.return_value.maybe_start.assert_called_once()
        assert init.call_args.args[0] == config
        worker_init.shutdown_telemetry_safely()
        shutdown.assert_called_once()
        collector.return_value.close.assert_called_once()


def test_pipeline_exports_once_and_keeps_early_gauge_callbacks(monkeypatch) -> None:
    import logging

    from opentelemetry import metrics, trace
    from opentelemetry._logs import _internal as logs
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.util._once import Once

    from twinkle.server.task_queue.worker import logger as task_logger
    from twinkle.server.telemetry import provider
    from twinkle.server.telemetry.metrics import MetricsRegistry

    for module, name in ((trace, 'TRACER'), (metrics._internal, 'METER'), (logs, 'LOGGER')):
        monkeypatch.setattr(module, f'_{name}_PROVIDER_SET_ONCE', Once())
        monkeypatch.setattr(module, f'_{name}_PROVIDER', None)
    monkeypatch.setattr(metrics._internal, '_PROXY_METER_PROVIDER', metrics._internal._ProxyMeterProvider())
    monkeypatch.setattr(provider, '_shutdown', False)
    monkeypatch.setattr(MetricsRegistry, '_instance', None)
    early = MetricsRegistry.get()

    sink = InMemoryLogRecordExporter()
    reader = InMemoryMetricReader()
    monkeypatch.setattr(provider, 'OTLPLogExporter', lambda **_: sink)
    monkeypatch.setattr(provider, 'PeriodicExportingMetricReader', lambda *_, **__: reader)

    config = TelemetryConfig(
        enabled=True, service_name='audit', export_interval_ms=100,
        resource_attributes={'service.instance.id': 'shared', 'deployment.environment.name': 'test'},
    )
    try:
        assert provider.init_telemetry(config)
        assert provider.init_telemetry(config)
        assert MetricsRegistry.get() is early
        early.set_resource_count('active_sessions', 7)
        early.request_duration_seconds.record(0.08)
        resource, = reader.get_metrics_data().resource_metrics
        assert resource.resource.attributes['deployment.environment.name'] == 'test'
        assert resource.resource.attributes['service.instance.id'] != 'shared'
        data = {m.name: m.data.data_points for scope in resource.scope_metrics for m in scope.metrics}
        assert [p.value for p in data['twinkle.sessions.active']] == [7]
        assert [p.value for p in data['twinkle.telemetry.heartbeat']] == [1]
        bounds = data['twinkle.http.request.duration_seconds'][0].explicit_bounds
        assert 0.05 in bounds and 0.1 in bounds

        root = logging.getLogger('audit')
        with traced_operation('log.audit') as span:
            context = span.get_span_context()
            root.warning('stdlib audit')
            logging.getLogger('twinkle').warning('twinkle audit')
            task_logger.info('task audit', extra={'twinkle_task_id': 'req-audit'})
            logging.getLogger('opentelemetry.audit').warning('transport audit')
            try:
                raise ValueError('private error text')
            except ValueError:
                root.exception('private error text')
        provider._logger_provider.force_flush()
        exported = sink.get_finished_logs()
        records = [entry.log_record for entry in exported]
        for body in ('stdlib audit', 'twinkle audit', 'task audit'):
            record, = [r for r in records if r.body == body]
            assert record.trace_id == context.trace_id
            assert record.span_id == context.span_id
        task_entry, = [entry for entry in exported if entry.log_record.body == 'task audit']
        assert task_entry.instrumentation_scope.name == 'twinkle.server.task_queue.worker'
        assert task_entry.log_record.attributes['twinkle_task_id'] == 'req-audit'
        assert not any(r.body == 'transport audit' for r in records)
        failure, = [r for r in records if r.attributes.get('error_type') == 'ValueError']
        assert failure.body == 'Operation failed'
        assert 'private error text' not in repr(dict(failure.attributes))
        handler = provider._logging_handler
    finally:
        provider.shutdown_telemetry()
    assert all(handler not in logging.getLogger(name).handlers for name in ('', 'twinkle'))
    assert not provider.init_telemetry(config)
    assert provider._logging_handler is None


def test_telemetry_export_interval_must_be_positive() -> None:
    from pydantic import ValidationError


    with pytest.raises(ValidationError, match='export_interval_ms'):
        TelemetryConfig(export_interval_ms=0)


# ---------- inbound HTTP trace-context continuity ------------------ #


def test_inbound_traceparent_continues_trace() -> None:
    """The tracing middleware extracts the inbound ``traceparent`` and starts
    its SERVER span within that context, so a Gateway -> Model / Gateway ->
    Sampler hop continues one trace instead of starting a fresh one.

    The active span's ``trace_id`` is captured inside the request handler (it
    runs nested inside the middleware's SERVER span), which makes the assertion
    independent of whichever global tracer provider another test may have
    installed.
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
    the handler under a span on a fresh trace."""
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
