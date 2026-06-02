#!/usr/bin/env python
"""Drive a running mock Twinkle server with enough traffic that every panel on
the ``Twinkle Server Overview`` Grafana dashboard shows data.

Why this script exists
----------------------
Most empty panels on the overview dashboard are NOT a wiring problem — they
are caused by:

1.  ``histogram_quantile(0.95, rate(..._bucket[5m]))`` returns NaN when the
    5-minute look-back has zero traffic, so latency / wait-time / execution
    panels read "No data".
2.  ``up_down_counter`` gauges emit on *delta* only. ``active_sessions`` /
    ``active_models`` / ``queue_depth`` stay invisible until something
    actually changes their underlying count at least once.
3.  Mock backends execute in microseconds — even when traffic exists,
    histogram P95 hugs the bottom bucket. Bump ``--max-tokens`` so the
    sampler's per-request runtime lifts off the floor.

Pre-reqs
--------
1.  Observability stack + Redis running::

        docker compose -f cookbook/observability/docker-compose.yaml up -d
        docker run -d --name twinkle-redis -p 6379:6379 redis:7

2.  Mock server running with telemetry **enabled** and a SHARED persistence
    backend. The shipped ``cookbook/client/server/mock/server_config.yaml``
    ships ``mode: memory`` which is per-process — Gateway-worker sessions
    are invisible to Model worker, the adapter countdown loop sees "session
    not found" and expires registered adapters within ~10s (which empties
    the ``twinkle_models_active`` gauge). For this load script to populate
    every panel, switch persistence to redis::

        persistence: { mode: redis, redis_url: redis://localhost:6379 }

    Telemetry: the worker reads the env flag as the literal string ``"1"``
    (see ``telemetry.worker_init.ensure_telemetry_initialized``), NOT
    ``"true"``::

        TWINKLE_TELEMETRY_ENABLED=1 \\
        TWINKLE_TELEMETRY_ENDPOINT=http://localhost:4317 \\
        python -m twinkle.server launch \\
            --config cookbook/client/server/mock/server_config.yaml

    Start Ray first (the launcher does ``ray.init(address='auto')`` and
    will refuse to spin one up locally)::

        ray start --head --num-cpus=4 --disable-usage-stats

Usage
-----
::

    # Defaults: 4 concurrent users, 120s, ~2 req/s each
    python cookbook/observability/load.py

    # Heavier: 8 users, 5 minutes, longer sampler runtime (lifts P95)
    python cookbook/observability/load.py \\
        --concurrency 8 --duration 300 --max-tokens 128

In Grafana set the time window to ``Last 15 minutes`` for the rate[5m]
queries to be meaningful.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid

import httpx

# Routes from cookbook/client/server/mock/server_config.yaml — keep in sync.
GATEWAY_ROUTE = '/api/v1'
MODEL_ROUTE = '/api/v1/model/mock'
SAMPLER_ROUTE = '/api/v1/sampler/mock'

# Any non-empty token is accepted (``is_token_valid`` is permissive by default).
TOKEN = 'load-test-token'


def _headers(session_id: str, *, request_id: str, multiplex_key: str | None = None) -> dict[str, str]:
    """Build the per-request header set the server middleware expects.

    The server's ``verify_request_token`` middleware requires:
    - ``Twinkle-Authorization: Bearer <token>``
    - ``X-Ray-Serve-Request-Id`` for sticky routing (any unique string ok)
    - ``X-Twinkle-Session-Id`` for session correlation (optional)

    Model + sampler deployments additionally call
    ``serve.get_multiplexed_model_id()`` for sticky-LoRA replica routing —
    Ray Serve raises ``ValueError("The model ID cannot be empty.")`` if the
    ``serve_multiplexed_model_id`` header is absent. Always set
    ``multiplex_key`` for model / sampler calls; the Gateway endpoint
    (``/api/v1/twinkle/create_session``) does not need it.

    Pass the SAME ``request_id`` for every call against the same registered
    adapter so the sticky-LoRA key (``request_id + '-' + adapter_name``)
    resolves to the registered resource on subsequent ``/forward_only`` calls.
    """
    headers = {
        'Twinkle-Authorization': f'Bearer {TOKEN}',
        'X-Ray-Serve-Request-Id': request_id,
        'X-Twinkle-Session-Id': session_id,
        'Content-Type': 'application/json',
    }
    if multiplex_key is not None:
        headers['serve_multiplexed_model_id'] = multiplex_key
    return headers


def _lora_config_payload(rank: int = 8) -> str:
    """JSON payload the server's ``deserialize_object`` will rehydrate into a
    ``peft.LoraConfig``. Matches ``twinkle_client.common.serialize.serialize_object``.
    """
    return json.dumps({
        '_TWINKLE_TYPE_': 'LoraConfig',
        'r': rank,
        'lora_alpha': rank * 2,
        'lora_dropout': 0.0,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
        'target_modules': ['q_proj', 'v_proj'],
    })


async def create_session(client: httpx.AsyncClient) -> str | None:
    """POST /api/v1/twinkle/create_session — returns the SERVER-issued
    ``session_id`` so subsequent ``X-Twinkle-Session-Id`` headers reference
    a session the server actually persisted. Using a client-side string
    would silently fail liveness checks (the adapter countdown loop in
    ``utils/lifecycle/base.py`` calls ``state.get_session_last_heartbeat``
    and expires adapters within ~10s when the ID isn't found)."""
    r = await client.post(
        f'{GATEWAY_ROUTE}/twinkle/create_session',
        headers=_headers('', request_id=uuid.uuid4().hex),
        json={'metadata': {'source': 'load.py'}},
        timeout=10.0,
    )
    if r.status_code != 200:
        print(f'  create_session -> {r.status_code} {r.text[:160]}')
        return None
    return r.json().get('session_id')


async def create_sampling_session(client: httpx.AsyncClient, session_id: str, model_path: str) -> None:
    """POST /api/v1/create_sampling_session — bumps ``active_sampling_sessions``.
    This is a Tinker route mounted at the gateway root (NOT under
    ``/twinkle/``); the Twinkle gateway handlers only expose ``create_session``."""
    try:
        await client.post(
            f'{GATEWAY_ROUTE}/create_sampling_session',
            headers=_headers(session_id, request_id=uuid.uuid4().hex),
            json={
                'session_id': session_id,
                'sampling_session_seq_id': 0,
                'model_path': model_path,
                'base_model': 'mock-model',
            },
            timeout=10.0,
        )
    except Exception:
        pass


async def session_heartbeat(client: httpx.AsyncClient, session_id: str) -> None:
    """POST /api/v1/twinkle/session_heartbeat — refreshes the session so
    the adapter countdown loop doesn't expire registered adapters mid-load."""
    try:
        await client.post(
            f'{GATEWAY_ROUTE}/twinkle/session_heartbeat',
            headers=_headers(session_id, request_id=uuid.uuid4().hex),
            json={'session_id': session_id},
            timeout=5.0,
        )
    except Exception:
        pass


async def add_adapter(client: httpx.AsyncClient, adapter_name: str, session_id: str, request_id: str) -> bool:
    """POST /api/v1/model/mock/twinkle/add_adapter_to_model — moves
    ``active_models`` gauge and goes through the task queue (queue_depth +
    task_execution histograms).
    """
    body = {'adapter_name': adapter_name, 'config': _lora_config_payload()}
    r = await client.post(
        f'{MODEL_ROUTE}/twinkle/add_adapter_to_model',
        headers=_headers(session_id, request_id=request_id, multiplex_key=adapter_name),
        json=body,
        timeout=30.0,
    )
    if r.status_code != 200:
        print(f'  add_adapter {adapter_name} -> {r.status_code} {r.text[:200]}')
        return False
    return True


async def sample(client: httpx.AsyncClient, session_id: str, *, max_tokens: int) -> int:
    """POST /api/v1/sampler/mock/twinkle/sample — primary load. No adapter
    registration needed (``adapter_name=''`` skips the resource check)."""
    body = {
        'inputs': [{'input_ids': [random.randint(0, 100) for _ in range(8)]}],
        'sampling_params': {'max_tokens': max_tokens},
        'adapter_name': '',
    }
    r = await client.post(
        f'{SAMPLER_ROUTE}/twinkle/sample',
        headers=_headers(session_id, request_id=uuid.uuid4().hex, multiplex_key=session_id),
        json=body,
        timeout=60.0,
    )
    return r.status_code


async def forward_only(client: httpx.AsyncClient, adapter_name: str, session_id: str, request_id: str) -> int:
    """POST /api/v1/model/mock/twinkle/forward_only against a registered adapter.

    ``request_id`` MUST be the same one used by the original ``add_adapter``
    call — the server prefixes ``request_id`` onto the adapter key for
    sticky-LoRA routing, so reusing it lets ``assert_resource_exists`` find
    the adapter we registered.
    """
    body = {
        'inputs': [{'input_ids': [random.randint(0, 100) for _ in range(16)]}],
        'adapter_name': adapter_name,
    }
    r = await client.post(
        f'{MODEL_ROUTE}/twinkle/forward_only',
        headers=_headers(session_id, request_id=request_id, multiplex_key=adapter_name),
        json=body,
        timeout=30.0,
    )
    return r.status_code


async def user_loop(
    user_id: int,
    base_url: str,
    deadline: float,
    interval: float,
    max_tokens: int,
) -> None:
    """Per-user driver: create_session + add_adapter once, then loop sample
    (and occasional forward_only) until the deadline. Periodically heartbeats
    the session so the adapter countdown loop doesn't expire registered
    adapters mid-load (default adapter_timeout is 1800s, but a missing
    heartbeat trips ``_is_session_alive`` long before that)."""
    adapter_name = f'adapter-u{user_id}-{uuid.uuid4().hex[:6]}'
    sticky_request_id = uuid.uuid4().hex
    heartbeat_interval = 5.0

    async with httpx.AsyncClient(base_url=base_url) as client:
        # IMPORTANT: use the SERVER-issued session_id; sending our own client-
        # side string would never match a stored session and registered
        # adapters would expire within ~10s.
        session_id = await create_session(client)
        adapter_ok = False
        if session_id:
            adapter_ok = await add_adapter(client, adapter_name, session_id, sticky_request_id)
            # Best-effort: bump the active_sampling_sessions gauge.
            await create_sampling_session(client, session_id, model_path=f'mock://{adapter_name}')

        ok_n = err_n = 0
        last_hb = time.monotonic()
        while time.monotonic() < deadline:
            if session_id and time.monotonic() - last_hb >= heartbeat_interval:
                await session_heartbeat(client, session_id)
                last_hb = time.monotonic()
            use_forward = adapter_ok and random.random() < 0.2
            try:
                if use_forward:
                    status = await forward_only(client, adapter_name, session_id, sticky_request_id)
                else:
                    status = await sample(client, session_id or '', max_tokens=max_tokens)
            except Exception as exc:
                err_n += 1
                print(f'  user {user_id} request error: {exc!r}')
                await asyncio.sleep(1.0)
                continue
            if 200 <= status < 300:
                ok_n += 1
            else:
                err_n += 1
            await asyncio.sleep(max(0.01, interval + random.uniform(-interval / 4, interval / 4)))

        print(f'  user {user_id:>2}  ok={ok_n:>4}  err={err_n}  '
              f'session={session_id or "<none>"}  adapter={adapter_name if adapter_ok else "<skip>"}')


async def main_async(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.duration
    print(f'Load:  base={args.base_url}  concurrency={args.concurrency}  '
          f'duration={args.duration}s  interval={args.interval}s  max_tokens={args.max_tokens}')
    print(f'Hits:  POST {GATEWAY_ROUTE}/twinkle/create_session')
    print(f'       POST {MODEL_ROUTE}/twinkle/add_adapter_to_model')
    print(f'       POST {MODEL_ROUTE}/twinkle/forward_only (~20%)')
    print(f'       POST {SAMPLER_ROUTE}/twinkle/sample     (~80%)')
    await asyncio.gather(*[
        user_loop(i, args.base_url, deadline, args.interval, args.max_tokens) for i in range(args.concurrency)
    ])
    print('Done. Allow ~30s for the next OTLP export tick, then refresh Grafana ')
    print('with the time window set to "Last 15 minutes".')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base-url', default='http://localhost:8000', help='Server base URL (default: %(default)s)')
    p.add_argument('--concurrency', type=int, default=4, help='Parallel users (default: %(default)s)')
    p.add_argument('--duration', type=int, default=120, help='Total seconds to run (default: %(default)s)')
    p.add_argument(
        '--interval',
        type=float,
        default=0.5,
        help='Mean seconds between requests per worker; ±25%% jitter applied. '
        'Lower → higher RPS. Default: %(default)s')
    p.add_argument(
        '--max-tokens',
        type=int,
        default=64,
        help='Mock sampler runtime scales with max_tokens. Bump to >= 64 so '
        'task_execution P95 lifts off the bottom histogram bucket. '
        'Default: %(default)s')
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
