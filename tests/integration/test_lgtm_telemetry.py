# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end OTLP telemetry tests against a local backend (R11.x, R13.3).

Pushes traces via OTLP and reads them back through the trace backend's
HTTP API to verify:

- correlation keys land on business spans (R11.2)
- the trace-context carrier round-trip places gateway/model/sampler spans
  under one trace id, even across the OTLP pipeline (R13.3)

The test auto-detects which trace backend is reachable on
``http://localhost:4317`` (OTLP gRPC):

* **Tempo via Grafana** at ``http://localhost:3000`` — preferred. Bring
  it up with the bundled stack: ``docker compose -f
  cookbook/observability/docker-compose.yaml up -d``.
* **Jaeger** at ``http://localhost:16686`` — lighter fallback with the
  same OTLP receiver. Start with ``docker run -d -e COLLECTOR_OTLP_ENABLED=true
  -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:1.62.0``.

Skips when neither is up.

Resource-metric exposure (R12.1) and Grafana dashboard structure (R12.5)
are already covered by the in-process tests in
``tests/server/telemetry/test_tracing_and_correlation.py``; the OTLP-→-Mimir
hop is OTel SDK code, not Twinkle code, so it has no separate Twinkle test.
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
JAEGER_URL = os.environ.get('TWINKLE_TEST_JAEGER_URL', 'http://localhost:16686')


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
        return httpx.get(f'{GRAFANA_URL}/api/health', timeout=2.0).status_code == 200
    except Exception:
        return False


def _jaeger_ready() -> bool:
    if not _tcp_open(JAEGER_URL):
        return False
    try:
        return httpx.get(f'{JAEGER_URL}/', timeout=2.0).status_code == 200
    except Exception:
        return False


def _detect_backend() -> str | None:
    if not _tcp_open(OTLP_ENDPOINT):
        return None
    if _grafana_ready():
        return 'tempo'
    if _jaeger_ready():
        return 'jaeger'
    return None


_BACKEND = _detect_backend()

pytestmark = pytest.mark.skipif(
    _BACKEND is None,
    reason=(
        f'No trace backend reachable. OTLP at {OTLP_ENDPOINT}, Grafana at {GRAFANA_URL}, '
        f'Jaeger at {JAEGER_URL}. Start one (cookbook/observability/docker-compose.yaml '
        'or `docker run jaegertracing/all-in-one:1.62.0`).'
    ),
)


# ---------- helpers ------------------------------------------------------- #


def _force_replace_global_providers(tracer_provider, meter_provider) -> None:
    """Force-replace the global OTel providers even if another test already set them.

    OTel's ``set_tracer_provider`` is one-shot per process — the conftest in
    ``tests/server/telemetry/`` may have installed an in-memory exporter that
    we'd otherwise inherit. Reset the underlying ``_TRACER_PROVIDER_SET_ONCE``
    guard so OTLP exporters become active for these tests.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.util._once import Once

    # Replace tracer provider.
    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(tracer_provider)

    # Replace meter provider.
    metrics._METER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    metrics._METER_PROVIDER = None  # type: ignore[attr-defined]
    metrics.set_meter_provider(meter_provider)


@contextmanager
def _telemetry_session(service_name: str):
    """Initialize a fresh OTLP pipeline pointed at the local backend, force-flush at exit."""
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({'service.name': service_name})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT)))

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT),
        export_interval_millis=1000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    _force_replace_global_providers(tracer_provider, meter_provider)
    try:
        yield service_name
    finally:
        try:
            tracer_provider.force_flush(timeout_millis=5000)
            meter_provider.force_flush(timeout_millis=5000)
        except Exception:
            pass


def _query_trace(service: str, trace_id_hex: str, attempts: int = 30, delay: float = 1.0) -> dict | None:
    """Poll the configured backend until ``trace_id_hex`` appears."""
    if _BACKEND == 'tempo':
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

    # Jaeger: GET /api/traces/{id}
    url = f'{JAEGER_URL}/api/traces/{trace_id_hex}'
    for _ in range(attempts):
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code == 200:
                data = r.json().get('data') or []
                if data and data[0].get('spans'):
                    return data[0]
        except Exception:
            pass
        time.sleep(delay)
    return None


def _spans_in_trace(payload: dict) -> list[dict]:
    """Return a normalized list of spans across both backends."""
    if _BACKEND == 'tempo':
        out = []
        for batch in payload.get('batches', []):
            for scope in batch.get('scopeSpans', []):
                for span in scope.get('spans', []):
                    out.append(
                        {
                            'name': span.get('name'),
                            'attributes': {
                                a['key']: a.get('value', {}).get('stringValue')
                                for a in span.get('attributes', [])
                            },
                        }
                    )
        return out
    # Jaeger trace JSON: top-level "spans" with operationName + tags.
    return [
        {
            'name': s['operationName'],
            'attributes': {t['key']: t.get('value') for t in s.get('tags', [])},
        }
        for s in payload.get('spans', [])
    ]


# ---------- 7.15: trace + correlation visible in the trace store --------- #


def test_business_span_with_correlation_visible_e2e() -> None:
    """A business span carrying twinkle.session_id / twinkle.model_id is
    retrievable from the trace store after going through the OTLP pipeline
    (R11.2)."""
    from opentelemetry import trace

    from twinkle.server.telemetry.correlation import MODEL_ID, SESSION_ID
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

    payload = _query_trace(service, trace_id_hex)
    assert payload is not None, f'trace {trace_id_hex} not found in {_BACKEND}'

    attrs_per_span = [s['attributes'] for s in _spans_in_trace(payload)]
    assert any(a.get(SESSION_ID) == session_id for a in attrs_per_span), (
        f'{SESSION_ID} not on any span in {_BACKEND}: {attrs_per_span}'
    )
    assert any(a.get(MODEL_ID) == model_id for a in attrs_per_span), (
        f'{MODEL_ID} not on any span in {_BACKEND}: {attrs_per_span}'
    )


# ---------- 10.4: single-trace-id fan-out across deployments (R13.3) ----- #


def test_carrier_round_trip_shares_trace_id_e2e() -> None:
    """Simulate the Gateway → Model → Sampler hop via the carrier helpers.
    The trace store records all three spans under one trace id."""
    from opentelemetry import trace

    from twinkle.server.telemetry.context_carrier import activate_carrier, make_carrier

    service = f'twinkle-test-fanout-{uuid.uuid4().hex[:6]}'
    with _telemetry_session(service):
        tracer = trace.get_tracer('twinkle.test.fanout')

        with tracer.start_as_current_span('gateway.route') as parent:
            trace_id = parent.get_span_context().trace_id
            carrier = make_carrier()

        with activate_carrier(carrier):
            with tracer.start_as_current_span('model.handle') as child:
                assert child.get_span_context().trace_id == trace_id
                downstream = make_carrier()

        with activate_carrier(downstream):
            with tracer.start_as_current_span('sampler.handle') as grandchild:
                assert grandchild.get_span_context().trace_id == trace_id

    trace_id_hex = format(trace_id, '032x')
    payload = _query_trace(service, trace_id_hex)
    assert payload is not None, f'fan-out trace {trace_id_hex} not found in {_BACKEND}'

    span_names = {s['name'] for s in _spans_in_trace(payload)}
    assert {'gateway.route', 'model.handle', 'sampler.handle'}.issubset(span_names), span_names
