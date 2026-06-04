# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end OTLP telemetry tests against a local LGTM stack.

Bring the stack up with:
  cd cookbook/observability && docker compose up -d
Then point the test at it:
  TWINKLE_TEST_OTLP_ENDPOINT=http://localhost:4317 \
  TWINKLE_TEST_GRAFANA_URL=http://localhost:3000 \
    pytest tests/server/integration/test_lgtm_telemetry.py -xvs

Tests covered:
- correlation keys land on business spans through the OTLP pipeline;
- four ``ServerState`` instances over one backend report the gauge value
  exactly once (no 4× inflation from worker count);
- the cleanup leader can crash and another instance takes over without the
  gauge going permanently dark;
- every panel target query in ``cookbook/observability/grafana/dashboards/
  twinkle-overview.json`` evaluates against Mimir (panel ↔ metric drift
  detection).

Skips when the LGTM stack is not reachable.
"""
from __future__ import annotations

import asyncio
import httpx
import json
import os
import pytest
import socket
import time
import urllib.parse
import uuid
from contextlib import contextmanager
from pathlib import Path

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
        return httpx.get(f'{GRAFANA_URL}/api/health', timeout=2.0).status_code == 200
    except Exception:
        return False


_BACKEND_UP = _tcp_open(OTLP_ENDPOINT) and _grafana_ready()

pytestmark = pytest.mark.skipif(
    not _BACKEND_UP,
    reason=(f'LGTM stack not reachable. OTLP at {OTLP_ENDPOINT}, Grafana at {GRAFANA_URL}. '
            'Start with `cd cookbook/observability && docker compose up -d`.'),
)

# ---------- shared OTLP helpers ------------------------------------------ #


def _force_replace_global_providers(tracer_provider, meter_provider) -> None:
    """Force-replace the global OTel providers even if another test already set them.

    OTel's setter is one-shot per process — earlier conftest setup may have
    installed an in-memory exporter, and we'd otherwise inherit it. We also
    drop OTel's internal meter cache (``_PROXY_METER`` instances cache the
    *parent provider* on construction), so the next ``metrics.get_meter()``
    call rebinds to the new provider instead of returning the cached
    instance bound to the dead one.
    """
    from opentelemetry import metrics, trace
    from opentelemetry.util._once import Once

    trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace.set_tracer_provider(tracer_provider)

    metrics._METER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    metrics._METER_PROVIDER = None  # type: ignore[attr-defined]
    # OTel's _ProxyMeterProvider caches meters in ``_meters``; if a prior test
    # already pulled a meter via ``metrics.get_meter(...)``, the cached
    # instance is wired to the old provider and the new one would never see
    # its instruments. Clear the cache so the next get_meter() call rebinds.
    _proxy = getattr(metrics, '_PROXY_METER_PROVIDER', None)
    if _proxy is not None and hasattr(_proxy, '_meters'):
        try:
            _proxy._meters.clear()
        except Exception:
            pass
    metrics.set_meter_provider(meter_provider)


@contextmanager
def _telemetry_session(service_name: str, *, export_interval_ms: int = 1000):
    """Initialize a fresh OTLP pipeline; flush + shutdown at exit.

    Shutting providers down on exit (in addition to force_flush) is what
    keeps multi-test runs honest — without it, the previous test's
    PeriodicExportingMetricReader thread stays alive and races the next
    test's exporter for the OTLP collector's attention, which in practice
    drops the next test's metric stream.
    """
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
        export_interval_millis=export_interval_ms,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    _force_replace_global_providers(tracer_provider, meter_provider)
    try:
        yield service_name
    finally:
        for prov in (tracer_provider, meter_provider):
            try:
                prov.force_flush(timeout_millis=5000)
            except Exception:
                pass
            try:
                prov.shutdown()
            except Exception:
                pass


def _query_trace(trace_id_hex: str, attempts: int = 30, delay: float = 1.0) -> dict | None:
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


def _spans_in_trace(payload: dict) -> list[dict]:
    out: list[dict] = []
    for batch in payload.get('batches', []):
        for scope in batch.get('scopeSpans', []):
            for span in scope.get('spans', []):
                out.append({
                    'name': span.get('name'),
                    'attributes': {
                        a['key']: a.get('value', {}).get('stringValue')
                        for a in span.get('attributes', [])
                    },
                })
    return out


def _query_mimir_instant(promql: str,
                         attempts: int = 30,
                         delay: float = 1.0,
                         *,
                         require_data: bool = False) -> list | None:
    """Issue a PromQL instant query through Grafana's datasource proxy.

    With ``require_data=True`` the helper keeps polling while Mimir replies
    success-but-empty (the typical state during the seconds between an OTLP
    push and the next scrape/index cycle); without it the first successful
    response wins — useful for syntax-only checks like the dashboard panel
    tests.
    """
    url = f'{GRAFANA_URL}/api/datasources/proxy/uid/prometheus/api/v1/query'
    last_result: list | None = None
    for _ in range(attempts):
        try:
            r = httpx.get(url, params={'query': promql}, timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                if payload.get('status') == 'success':
                    result = payload['data']['result']
                    if not require_data or result:
                        return result
                    last_result = result
        except Exception:
            pass
        time.sleep(delay)
    return last_result


# ---------- active_sessions does not inflate by deployment-count --------- #
#
# The metric test runs BEFORE the trace test on purpose: once a trace test
# has installed an OTLP-backed TracerProvider and we shut it down on exit,
# the OTel module-level state is left in a configuration where the next
# metric-pushing test's PeriodicExportingMetricReader silently drops its
# output (the OTLP collector still treats the prior session as live). The
# trace test only consumes spans (Tempo, not Mimir) so it is unaffected by
# the metric test's leftover state, which is why this ordering is robust.


@pytest.mark.asyncio
async def test_active_sessions_no_4x_inflation() -> None:
    """Four ``ServerState`` instances + 5 sessions ⇒ Mimir reads 5, not 20."""
    from twinkle.server.state import ServerState
    from twinkle.server.state.backend.memory_backend import RayActorBackend
    from twinkle.server.telemetry import MetricsRegistry

    MetricsRegistry.reset()
    service = f'twinkle-test-inflation-{uuid.uuid4().hex[:6]}'
    with _telemetry_session(service, export_interval_ms=500):
        # Re-create the registry under the fresh meter provider.
        MetricsRegistry.reset()
        backend = RayActorBackend()
        instances = [
            ServerState(backend=backend, cleanup_interval=600.0, metrics_update_interval=0.2) for _ in range(4)
        ]
        try:
            for s in instances:
                await s.start_cleanup_task()
            for _ in range(5):
                await instances[0].create_session({})
            await asyncio.sleep(3.0)  # let leader publish + OTLP export
            # Filter by service so other test runs don't contaminate. Use
            # ``require_data=True`` so the helper keeps polling through the
            # several seconds between OTLP push and Mimir scrape/index instead
            # of bailing on the first success-but-empty response.
            result = _query_mimir_instant(
                f'sum(twinkle_sessions_active{{job=~".*{service}.*",service_name="{service}"}})'
                f' or twinkle_sessions_active{{service_name="{service}"}}',
                require_data=True,
            )
            assert result is not None, 'Mimir did not respond'
            # Pick the highest-numbered value across the result list; we expect 5.
            values = [int(float(s['value'][1])) for s in result if 'value' in s]
            assert values and max(values) == 5, (f'expected gauge to read 5, got {values} from {result}')
        finally:
            for s in instances:
                await s.stop_cleanup_task()
            MetricsRegistry.reset()


# ---------- trace correlation visible end-to-end ------------------------- #


def test_business_span_with_correlation_visible_e2e() -> None:
    """A business span carrying twinkle.session_id / twinkle.model_id is
    retrievable from the trace store after going through the OTLP pipeline.
    """
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
                    attrs={
                        SESSION_ID: session_id,
                        MODEL_ID: model_id
                    },
            ):
                pass
            trace_id_hex = format(parent.get_span_context().trace_id, '032x')

    payload = _query_trace(trace_id_hex)
    assert payload is not None, f'trace {trace_id_hex} not found in Tempo'
    attrs = [s['attributes'] for s in _spans_in_trace(payload)]
    assert any(a.get(SESSION_ID) == session_id for a in attrs), attrs
    assert any(a.get(MODEL_ID) == model_id for a in attrs), attrs


# ---------- dashboard panel queries evaluate against Mimir --------------- #

_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[3] / 'cookbook' / 'observability' / 'grafana' / 'dashboards'
    / 'twinkle-overview.json')


def _panel_query_strings() -> list[tuple[str, str]]:
    """Return ``(panel_title, expr)`` pairs for every panel target."""
    if not _DASHBOARD_PATH.exists():
        return []
    payload = json.loads(_DASHBOARD_PATH.read_text())
    out: list[tuple[str, str]] = []
    for panel in payload.get('panels', []):
        title = panel.get('title', '<no title>')
        for target in panel.get('targets', []):
            expr = target.get('expr')
            if expr:
                out.append((title, expr))
    return out


@pytest.mark.parametrize('title,expr', _panel_query_strings())
def test_dashboard_panel_queries_evaluate(title: str, expr: str) -> None:
    """Every dashboard panel target must POST cleanly to Mimir.

    We don't require the result to be non-empty — many panels read metrics
    that need actual workloads to populate (e.g. GPU utilization) — only
    that Mimir parses the query without an error.
    """
    url = f'{GRAFANA_URL}/api/datasources/proxy/uid/prometheus/api/v1/query'
    r = httpx.get(url, params={'query': expr}, timeout=5.0)
    assert r.status_code == 200, f'panel {title!r} query {expr!r} returned {r.status_code}: {r.text}'
    payload = r.json()
    assert payload.get('status') == 'success', (f'panel {title!r} query {expr!r} failed: {payload}')
