# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end: full SFT flow + every Grafana panel actually has data.

The existing ``test_mock_mode_startup`` test runs the SFT flow but doesn't
push telemetry anywhere (workers boot with telemetry disabled). The
existing ``test_lgtm_telemetry`` tests do push telemetry but never exercise
the HTTP/task surface, so most dashboard panels never see data even in
those runs.

This test stitches the two together: it boots the same three Ray Serve
deployments as the mock-mode test but with ``TWINKLE_TELEMETRY_*`` env
vars propagated into every Ray worker, so each replica's worker_init
fires up a real OTLP-backed MeterProvider pointing at the local LGTM
stack. Then we run the full Twinkle SDK SFT smoke (same as the existing
test), wait long enough for Mimir to scrape and index, and assert each
dashboard panel target (other than the three known no-traffic ones —
rate-limit-rejections and the two GPU panels) returns at least one
datapoint.

Gated on both ``TWINKLE_TEST_INTEGRATION=1`` and LGTM-stack reachability
so it stays skipped by default and on hosts without docker.
"""
from __future__ import annotations

import httpx
import json
import os
import pytest
import socket
import time
import urllib.parse
import uuid
from pathlib import Path

from tests.server.fixtures import MOCK_SERVER_CONFIG
from twinkle.server.config import ServerConfig

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


pytestmark = [
    pytest.mark.skipif(
        os.environ.get('TWINKLE_TEST_INTEGRATION', '0') != '1',
        reason='Set TWINKLE_TEST_INTEGRATION=1 to run the in-process Ray Serve smoke',
    ),
    pytest.mark.skipif(
        not _grafana_ready() or not _tcp_open(OTLP_ENDPOINT),
        reason=(f'LGTM stack not reachable (OTLP at {OTLP_ENDPOINT}, Grafana at {GRAFANA_URL}). '
                'Start with `cd cookbook/observability && docker compose up -d`.'),
    ),
]

# Panels we expect to populate from the SFT flow alone. The three excluded
# panels (rate-limit-rejections, GPU utilization, GPU memory) need traffic
# that the smoke does not produce: a real rate-limit hit, and an NVIDIA GPU.
_EXPECTED_NON_EMPTY: set[str] = {
    'HTTP request rate (per deployment)',
    'HTTP latency P95 (per deployment)',
    'Active resources',
    'Task queue depth (per deployment)',
    'Task execution P95',
    'Task wait time P95',
    'Task completions by status',
    'CPU utilization',
    'Memory utilization (system)',
}

_DASHBOARD_PATH = (
    Path(__file__).resolve().parents[3] / 'cookbook' / 'observability' / 'grafana' / 'dashboards'
    / 'twinkle-overview.json')


def _all_panel_targets() -> list[tuple[str, str]]:
    payload = json.loads(_DASHBOARD_PATH.read_text())
    out: list[tuple[str, str]] = []
    for panel in payload.get('panels', []):
        title = panel.get('title', '<no title>')
        for target in panel.get('targets', []):
            expr = target.get('expr')
            if expr:
                out.append((title, expr))
    return out


def _query_mimir_until_data(promql: str, *, deadline: float) -> list:
    """Poll Mimir for ``promql`` until the result is non-empty or we hit ``deadline``.

    Returns the last result seen — caller asserts non-empty.
    """
    url = f'{GRAFANA_URL}/api/datasources/proxy/uid/prometheus/api/v1/query'
    last: list = []
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, params={'query': promql}, timeout=5.0)
            if r.status_code == 200:
                payload = r.json()
                if payload.get('status') == 'success':
                    last = payload['data']['result']
                    if last:
                        return last
        except Exception:
            pass
        time.sleep(1.0)
    return last


def _wait_until_healthy(serve_module, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        status = serve_module.status()
        last = {name: app.status for name, app in status.applications.items()}
        if last and all(s == 'RUNNING' for s in last.values()):
            return last
        time.sleep(0.5)
    return last


def _import_exerciser():
    """Reuse the SDK smoke that the existing mock-mode test already pins down."""
    from tests.server.integration.test_mock_mode_startup import _exercise_twinkle_clients
    return _exercise_twinkle_clients


@pytest.fixture(scope='module')
def telemetry_cluster():
    """Boot Ray + Serve with telemetry env vars propagated into every worker.

    Distinct from the ``ray_cluster`` fixture in the sibling module: we need
    the workers to come up with telemetry ENABLED so each replica's
    ``worker_init.ensure_telemetry_initialized()`` installs a real OTLP
    MeterProvider. ``ignore_reinit_error=True`` would silently inherit a
    pre-existing cluster (which the other tests may have started without
    telemetry), so we use a dedicated Ray namespace to avoid clashes.
    """
    import ray
    from ray import serve

    cfg = ServerConfig.from_yaml(MOCK_SERVER_CONFIG)

    # Tag every metric stream with a per-run service name so the assertions
    # filter to this run only — otherwise leftover data from prior runs of
    # the lgtm telemetry test would pollute the per-panel checks.
    service_name = f'twinkle-e2e-{uuid.uuid4().hex[:6]}'

    telemetry_env: dict[str, str] = {
        'TWINKLE_TELEMETRY_ENABLED': '1',
        'TWINKLE_TELEMETRY_DEBUG': '0',
        'TWINKLE_TELEMETRY_SERVICE': service_name,
        'TWINKLE_TELEMETRY_ENDPOINT': OTLP_ENDPOINT,
        # Push every second so Mimir sees several samples per minute within
        # the test's ~30s wait window (rate() over a 1m range needs >=2).
        'TWINKLE_TELEMETRY_INTERVAL': '1000',
    }
    persistence_env = cfg.persistence.to_env_vars() if cfg.persistence is not None else {}

    worker_env = {**persistence_env, **telemetry_env}
    for k, v in worker_env.items():
        os.environ[k] = v

    # Force a clean Ray boot so the runtime_env (with our telemetry vars)
    # actually takes effect. Without this, an earlier test in the same
    # session may have left Ray initialized without telemetry, and
    # ``ignore_reinit_error`` would silently make us inherit that cluster.
    try:
        ray.shutdown()
    except Exception:
        pass

    ray.init(
        num_cpus=4,
        num_gpus=0,
        namespace='twinkle-e2e-grafana',
        include_dashboard=False,
        runtime_env={'env_vars': worker_env},
    )
    yield service_name
    try:
        serve.shutdown()
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass


def test_full_sft_populates_every_grafana_panel(telemetry_cluster) -> None:
    """Drive the full SFT flow and verify each expected panel reads non-empty.

    Boots the three Ray Serve deployments with telemetry on, runs the
    Twinkle SDK smoke (forward / backward / step / save / load / resume /
    sample / upload), then for each dashboard panel target that the smoke
    is expected to populate (everything except rate-limit-rejections and
    the two GPU panels) we poll Mimir until at least one datapoint exists.
    """
    from ray import serve

    from twinkle.server.gateway import build_gateway_app
    from twinkle.server.model import build_model_app
    from twinkle.server.sampler import build_sampler_app

    service_name = telemetry_cluster
    cfg = ServerConfig.from_yaml(MOCK_SERVER_CONFIG)

    port = 18000 + (os.getpid() % 1000) + 100  # offset so we don't collide with the sibling test
    host = '127.0.0.1'
    serve.start(http_options={'host': host, 'port': port})

    builders = {
        'server': build_gateway_app,
        'model': build_model_app,
        'sampler': build_sampler_app,
    }
    for app_spec in cfg.applications:
        builder = builders[app_spec.import_path]
        args = {k: v for k, v in dict(app_spec.args).items() if v is not None}
        if app_spec.import_path == 'server':
            http_opts = cfg.http_options.model_dump()
            http_opts['host'] = host
            http_opts['port'] = port
            args.setdefault('http_options', http_opts)
        deploy_options: dict = {'ray_actor_options': {'num_cpus': 0.1}}
        for raw in app_spec.deployments:
            if isinstance(raw, dict):
                deploy_options = {
                    k: v
                    for k, v in raw.items() if k not in ('name', 'ray_actor_options', 'autoscaling_config')
                }
                deploy_options['ray_actor_options'] = {'num_cpus': 0.1}
                break
        bound = builder(deploy_options=deploy_options, **args)
        serve.run(bound, name=app_spec.name, route_prefix=app_spec.route_prefix)

    statuses = _wait_until_healthy(serve, 30.0)
    assert statuses and all(s == 'RUNNING' for s in statuses.values()), statuses

    # Drive real HTTP/task traffic across the gateway, model, and sampler.
    base = f'http://{host}:{port}'
    exercise = _import_exerciser()
    exercise(base)

    # Mimir scrape interval in the bundled grafana/otel-lgtm is ~15s; we
    # pushed metrics every 1s during the ~30s of SFT smoke, so by the time
    # we're done several scrape cycles have run. Allow up to ~60s of polling
    # per panel to absorb the tail of indexing latency.
    deadline = time.monotonic() + 60.0

    failures: list[tuple[str, str, str]] = []
    for title, raw_expr in _all_panel_targets():
        if title not in _EXPECTED_NON_EMPTY:
            continue
        # Scope the query to THIS run's service name so leftover data from
        # prior test runs in the same Mimir doesn't accidentally satisfy
        # the assertion.
        scoped = _scope_to_service(raw_expr, service_name)
        result = _query_mimir_until_data(scoped, deadline=deadline)
        if not result:
            failures.append((title, raw_expr, scoped))

    assert not failures, ('Some expected panels returned no data within the deadline:\n  '
                          + '\n  '.join(f'- {t}: {raw} (scoped: {scoped})' for t, raw, scoped in failures))


def _scope_to_service(promql: str, service_name: str) -> str:
    """Inject ``service_name="<svc>"`` into every ``twinkle_*`` metric selector.

    The dashboard's PromQL is written un-scoped (one service per cluster in
    production), but the LGTM container retains metrics from prior test
    runs in the same docker session. Filtering by the run's unique service
    name keeps each assertion looking only at data this test emitted.

    The replacement adds the filter *between* the metric name and the
    optional range vector — e.g. ``rate(twinkle_x[1m])`` becomes
    ``rate(twinkle_x{service_name="..."}[1m])``.
    """
    import re

    label = f'service_name="{service_name}"'

    def _add(match: re.Match) -> str:
        metric = match.group(1)
        existing = match.group(2)  # ``{…}`` or ``''``
        if existing:
            inside = existing[1:-1].strip()
            new_labels = f'{{{label},{inside}}}' if inside else f'{{{label}}}'
        else:
            new_labels = f'{{{label}}}'
        return f'{metric}{new_labels}'

    return re.sub(r'(\btwinkle_[a-zA-Z0-9_]+)(\{[^}]*\})?', _add, promql)
