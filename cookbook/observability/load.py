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
1.  Observability stack running::

        docker compose -f cookbook/observability/docker-compose.yaml up -d

2.  Mock server running with telemetry **enabled** (the shipped
    ``cookbook/client/server/mock/server_config.yaml`` has
    ``telemetry.enabled: false`` — flip it to ``true`` or override via env
    before launching)::

        TWINKLE_TELEMETRY_ENABLED=true \\
        python -m twinkle.server launch \\
            --config cookbook/client/server/mock/server_config.yaml

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


def _headers(session_id: str, *, request_id: str) -> dict[str, str]:
    """Build the per-request header set the server middleware expects.

    The server's ``verify_request_token`` middleware requires:
    - ``Twinkle-Authorization: Bearer <token>``
    - ``X-Ray-Serve-Request-Id`` for sticky routing (any unique string ok)
    - ``X-Twinkle-Session-Id`` for session correlation (optional)

    Pass the SAME ``request_id`` for every call against the same registered
    adapter so the sticky-LoRA key (``request_id + '-' + adapter_name``)
    resolves to the registered resource on subsequent ``/forward_only`` calls.
    """
    return {
        'Twinkle-Authorization': f'Bearer {TOKEN}',
        'X-Ray-Serve-Request-Id': request_id,
        'X-Twinkle-Session-Id': session_id,
        'Content-Type': 'application/json',
    }


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


async def create_session(client: httpx.AsyncClient, session_id: str) -> bool:
    """POST /api/v1/twinkle/create_session — moves ``active_sessions`` gauge."""
    r = await client.post(
        f'{GATEWAY_ROUTE}/twinkle/create_session',
        headers=_headers(session_id, request_id=uuid.uuid4().hex),
        json={'metadata': {'source': 'load.py'}},
        timeout=10.0,
    )
    if r.status_code != 200:
        print(f'  create_session -> {r.status_code} {r.text[:160]}')
        return False
    return True


async def add_adapter(client: httpx.AsyncClient, adapter_name: str, session_id: str, request_id: str) -> bool:
    """POST /api/v1/model/mock/twinkle/add_adapter_to_model — moves
    ``active_models`` gauge and goes through the task queue (queue_depth +
    task_execution histograms).
    """
    body = {'adapter_name': adapter_name, 'config': _lora_config_payload()}
    r = await client.post(
        f'{MODEL_ROUTE}/twinkle/add_adapter_to_model',
        headers=_headers(session_id, request_id=request_id),
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
        headers=_headers(session_id, request_id=uuid.uuid4().hex),
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
        headers=_headers(session_id, request_id=request_id),
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
    (and occasional forward_only) until the deadline."""
    session_id = f'load-user-{user_id}-{uuid.uuid4().hex[:6]}'
    adapter_name = f'adapter-u{user_id}-{uuid.uuid4().hex[:6]}'
    # Pinned per-user request id for the sticky-LoRA path (forward_only against
    # the adapter we register in this loop iteration).
    sticky_request_id = uuid.uuid4().hex

    async with httpx.AsyncClient(base_url=base_url) as client:
        sess_ok = await create_session(client, session_id)
        adapter_ok = False
        if sess_ok:
            adapter_ok = await add_adapter(client, adapter_name, session_id, sticky_request_id)

        ok_n = err_n = 0
        while time.monotonic() < deadline:
            # 80% sample (no adapter), 20% forward_only (uses registered adapter).
            use_forward = adapter_ok and random.random() < 0.2
            try:
                if use_forward:
                    status = await forward_only(client, adapter_name, session_id, sticky_request_id)
                else:
                    status = await sample(client, session_id, max_tokens=max_tokens)
            except Exception as exc:
                err_n += 1
                print(f'  user {user_id} request error: {exc!r}')
                await asyncio.sleep(1.0)
                continue
            if 200 <= status < 300:
                ok_n += 1
            else:
                err_n += 1
            # Jittered interval keeps requests from clumping into the same scrape window.
            await asyncio.sleep(max(0.01, interval + random.uniform(-interval / 4, interval / 4)))

        print(f'  user {user_id:>2}  ok={ok_n:>4}  err={err_n}  '
              f'session={session_id}  adapter={adapter_name if adapter_ok else "<skip>"}')


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
