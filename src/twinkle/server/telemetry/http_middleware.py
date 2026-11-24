# Copyright (c) ModelScope Contributors. All rights reserved.
"""FastAPI HTTP request-metrics middleware.

Split out of the former ``middleware.py``: the metric adapters and
containers live in ``metrics.py`` next to the ``MetricsRegistry``; this file holds only
the HTTP middleware factory, so a file named for HTTP middleware contains HTTP middleware
and the queue code no longer imports a module called ``middleware`` just to reach
``get_task_metrics``.
"""
from __future__ import annotations
import asyncio
import time
from collections.abc import Callable
from typing import Any

from twinkle.server.telemetry.metrics import METERED_TASK_TYPES, MetricsRegistry

_OPERATIONS = METERED_TASK_TYPES | {'asample', 'session_heartbeat', 'retrieve_future'}


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
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except asyncio.CancelledError:
            status = None
            raise
        finally:
            if status is not None:
                elapsed = time.monotonic() - start
                route_name = request.path_params.get('endpoint') or getattr(request.scope.get('route'), 'name', None)
                operation = route_name if route_name in _OPERATIONS else 'request'
                tags = {'deployment': deployment, 'method': request.method, 'operation': operation}
                if fingerprint := getattr(request.state, 'tenant_fingerprint', None):
                    tags['tenant_fingerprint'] = fingerprint
                if base_model := getattr(request.state, 'base_model', None):
                    tags['base_model'] = base_model
                m = MetricsRegistry.get().request_metrics(deployment)
                request_tags = {**tags, 'status': str(status)}
                if status == 429:
                    request_tags['response_origin'] = (
                        'forwarded' if getattr(request.state, 'forwarded_429', False) else 'local'
                    )
                m.requests_total.inc(tags=request_tags)
                m.request_duration_seconds.observe(elapsed, tags=tags)

    return metrics_middleware
