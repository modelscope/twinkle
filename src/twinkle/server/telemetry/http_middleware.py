# Copyright (c) ModelScope Contributors. All rights reserved.
"""FastAPI HTTP request-metrics middleware.

Split out of the former ``middleware.py``: the metric adapters and
containers live in ``metrics.py`` next to the ``MetricsRegistry``; this file holds only
the HTTP middleware factory, so a file named for HTTP middleware contains HTTP middleware
and the queue code no longer imports a module called ``middleware`` just to reach
``get_task_metrics``.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from twinkle.server.telemetry.metrics import MetricsRegistry


def create_metrics_middleware(deployment: str) -> Callable:
    """Return a FastAPI ``http`` middleware that records request metrics.

    Usage inside a ``build_*_app()`` function::

        from twinkle.server.telemetry.http_middleware import create_metrics_middleware
        from twinkle.server.telemetry.tracing import create_tracing_middleware

        app.middleware('http')(verify_token)
        app.middleware('http')(create_tracing_middleware("Model"))
        app.middleware('http')(create_metrics_middleware("Model"))   # outermost

    FastAPI executes middleware in LIFO order, so the **last** middleware
    registered is the outermost wrapper. Register metrics last so its
    latency observation covers the full request path including tracing
    overhead and authentication.
    """

    async def metrics_middleware(request: Any, call_next: Callable) -> Any:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        status = str(response.status_code)
        method = request.scope['route'].path if 'route' in request.scope else request.url.path
        m = MetricsRegistry.get().request_metrics(deployment)
        m.requests_total.inc(tags={
            'deployment': deployment,
            'method': method,
            'status': status,
        })
        m.request_duration_seconds.observe(
            elapsed, tags={
                'deployment': deployment,
                'method': method,
            })
        return response

    return metrics_middleware
