# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end LGTM telemetry tests (R11.x, R12.x, R13.3).

Sends traces and metrics to a real OTLP endpoint exposed by the
``grafana/otel-lgtm`` docker container, then queries Tempo / Mimir back
through Grafana's HTTP API to confirm round-trip behaviour:

- correlation keys filterable in Tempo (R11.2)
- ``ResourceMetricsCollector`` gauges visible in Mimir (R12.1)
- a single Gateway → Model → Sampler trace shares one trace id (R13.3)

The tests are skipped when ``http://localhost:4317`` (OTLP gRPC) and
``http://localhost:3000`` (Grafana) aren't both reachable. Bring the
stack up with ``docker compose -f cookbook/observability/docker-compose.yaml up -d``.
"""
from __future__ import annotations

import os
import socket
import time
import urllib.parse
import uuid
from contextlib import contextmanager

import httpx
import pytest

OTLP_ENDPOINT = os.environ.get('TWINKLE_TEST_OTLP_ENDPOINT', 'http://localhost:4317')
GRAFANA_URL = os.environ.get('TWINKLE_TEST_GRAFANA_URL', 'http://localhost:3000')


def _tcp_open(url: str, timeout: float = 1.0) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _grafana_ready() -> bool:
    if not _tcp_open(GRAFANA_URL):
        return False
    try:
        r = httpx.get(f'{GRAFANA_URL}/api/health', timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_tcp_open(OTLP_ENDPOINT) and _grafana_ready()),
    reason=(
        f'LGTM stack unreachable at {OTLP_ENDPOINT} / {GRAFANA_URL}. '
        'Start it with `docker compose -f cookbook/observability/docker-compose.yaml up -d`.'
    ),
)


# ---------- helpers ------------------------------------------------------- #


@contextmanager
def _telemetry_session(service_name: str):
    """Initialize a real OTLP pipeline pointed at the LGTM stack and shut it
    down at the end of the block. Spans + metrics emitted inside the block
    are exported to the local stack."""
    from twinkle.server.telemetry.provider import TelemetryConfig, init_telemetry

    cfg = TelemetryConfig(
        enabled=True,
        debug=False,
        service_name=service_name,
        otlp_endpoint=OTLP_ENDPOINT,
        export_interval_ms=1000,
    )
    init_telemetry(cfg)
    try:
        yield cfg
    finally:
        # Force-flush so spans/metrics actually land before the test queries.
        try:
            from opentelemetry import metrics, trace

            trace.get_tracer_provider().force_flush(timeout_millis=5000)
            metrics.get_meter_provider().force_flush(timeout_millis=5000)
        except Exception:
            pass


def _query_tempo(trace_id_hex: str, attempts: int = 30, delay: float = 1.0) -> dict | None:
    """Poll Tempo via Grafana's datasource proxy until the trace appears."""
    url = f'{GRAFANA_URL}/api/datasources/proxy/uid/tempo/api/traces/{trace_id_hex}'
    for _ in range(attempts):
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200 and r.json().get('batches'):
                return r.json()
        except Exception:
            pass
        time.sleep(delay)
    return None


def _query_mimir(metric_name: str, attempts: int = 30, delay: float = 1.0) -> bool:
    """Return True when Mimir reports at least one sample for ``metric_name``."""
    url = f'{GRAFANA_URL}/api/datasources/proxy/uid/prometheus/api/v1/query'
    for _ in range(attempts):
        try:
            r = httpx.get(url, params={'query': metric_name}, timeout=5.0)
            if r.status_code == 200:
                data = r.json().get('data', {})
                if data.get('result'):
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False


# ---------- 7.15: trace + correlation visible in Tempo (R11.2) ------------ #


def test_business_span_with_correlation_visible_in_tempo() -> None:
    from opentelemetry import trace
    from twinkle.server.telemetry.correlation import SESSION_ID, MODEL_ID
    from twinkle.server.telemetry.tracing import traced_operation

    service = f'twinkle-test-trace-{uuid.uuid4().hex[:6]}'
    session_id = f'sess-{uuid.uuid4().hex[:8]}'
    model_id = f'mid-{uuid.uuid4().hex[:8]}'

    with _telemetry_session(service):
        tracer = trace.get_tracer('twinkle.test.trace')
        with tracer.start_as_current_span('integration.parent') as parent:
            with traced_operation(
                'server_state.register_model',
                attrs={SESSION_ID: session_id, MODEL_ID: model_id},
            ):
                pass
            trace_id_hex = format(parent.get_span_context().trace_id, '032x')

    payload = _query_tempo(trace_id_hex)
    assert payload is not None, f'trace {trace_id_hex} not found in Tempo'

    # Walk every span and confirm the correlation attributes landed.
    found_session = found_model = False
    for batch in payload['batches']:
        for scope in batch.get('scopeSpans', []):
            for span in scope.get('spans', []):
                for attr in span.get('attributes', []):
                    key = attr.get('key')
                    val = attr.get('value', {}).get('stringValue')
                    if key == SESSION_ID and val == session_id:
                        found_session = True
                    if key == MODEL_ID and val == model_id:
                        found_model = True
    assert found_session, f'{SESSION_ID} not found on any span in trace {trace_id_hex}'
    assert found_model, f'{MODEL_ID} not found on any span in trace {trace_id_hex}'


# ---------- 7.15: resource metrics visible in Mimir (R12.1) --------------- #


def test_resource_metrics_visible_in_mimir() -> None:
    from twinkle.server.telemetry import resource_metrics

    if not resource_metrics._PSUTIL_AVAILABLE:
        pytest.skip('psutil not installed in test env — collector cannot emit')

    service = f'twinkle-test-metrics-{uuid.uuid4().hex[:6]}'
    with _telemetry_session(service):
        resource_metrics.reset_collector_for_tests()
        resource_metrics.get_collector().maybe_start()
        # Drive at least one observation cycle.
        time.sleep(2.0)

    # Prometheus naming flips dots to underscores.
    for metric_name in (
        'twinkle_system_cpu_utilization',
        'twinkle_system_memory_usage_bytes',
        'twinkle_process_memory_usage_bytes',
    ):
        assert _query_mimir(metric_name), f'{metric_name} not visible in Mimir'


# ---------- 7.15 graceful: pynvml absent → no GPU data, no error --------- #


def test_no_gpu_means_no_gpu_data_no_error() -> None:
    """When pynvml is missing or no GPU is present, the GPU gauges are
    simply absent — no exception, no panic (R12.3)."""
    from unittest import mock
    from twinkle.server.telemetry import resource_metrics

    with mock.patch.object(resource_metrics, '_PYNVML_AVAILABLE', False):
        resource_metrics.reset_collector_for_tests()
        collector = resource_metrics.ResourceMetricsCollector()
        # Must not raise even when pynvml is unavailable.
        collector.maybe_start()
        # No GPU gauges registered.
        assert all(not g.startswith('twinkle.gpu.') for g in collector.registered_gauges)


# ---------- 10.4: cross-deployment trace propagation via carrier (R13.3) - #


def test_carrier_round_trip_shares_trace_id_in_tempo() -> None:
    """Simulate the Gateway → Model → Sampler hop via the carrier helpers
    and verify Tempo records all three spans under one trace id."""
    from opentelemetry import trace
    from twinkle.server.telemetry.context_carrier import activate_carrier, make_carrier

    service = f'twinkle-test-fanout-{uuid.uuid4().hex[:6]}'
    with _telemetry_session(service):
        tracer = trace.get_tracer('twinkle.test.fanout')

        with tracer.start_as_current_span('gateway.route') as parent:
            trace_id = parent.get_span_context().trace_id
            carrier = make_carrier()
        # Receiving side (Model handler) attaches the carrier and starts a child.
        with activate_carrier(carrier):
            with tracer.start_as_current_span('model.handle') as child:
                assert child.get_span_context().trace_id == trace_id
                # Re-emit a carrier for the next hop (Model → Sampler).
                downstream = make_carrier()
        with activate_carrier(downstream):
            with tracer.start_as_current_span('sampler.handle') as grandchild:
                assert grandchild.get_span_context().trace_id == trace_id

    trace_id_hex = format(trace_id, '032x')
    payload = _query_tempo(trace_id_hex)
    assert payload is not None, f'fan-out trace {trace_id_hex} not found in Tempo'

    span_names = {
        span.get('name')
        for batch in payload['batches']
        for scope in batch.get('scopeSpans', [])
        for span in scope.get('spans', [])
    }
    assert {'gateway.route', 'model.handle', 'sampler.handle'}.issubset(span_names), span_names
