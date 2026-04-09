# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Central metrics module for Twinkle server observability.

Provides ray.util.metrics instruments that feed both the Ray Dashboard
(port 8265) and Prometheus (via /api/prometheus).

All metric names use the ``twinkle_`` prefix.  Metric instances are
cached per deployment to avoid duplicate registration.

Public entry-points:

* ``create_metrics_middleware(deployment)`` – FastAPI HTTP middleware
* ``get_task_metrics(deployment)``          – task-queue / rate-limit gauges
* ``get_resource_metrics()``               – ServerState resource gauges
"""
from __future__ import annotations

import time
from pydantic import BaseModel, ConfigDict
from ray.util.metrics import Counter, Gauge, Histogram
from typing import Any, Callable

from twinkle.utils.logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Histogram bucket boundaries (seconds) – shared by all histograms
# ---------------------------------------------------------------------------
_HISTOGRAM_BOUNDARIES = [
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
]

# ---------------------------------------------------------------------------
# Lazy caches – populated on first call per deployment / globally
# ---------------------------------------------------------------------------
_task_metrics_cache: dict[str, TaskMetrics] = {}
_resource_metrics_cache: ResourceMetrics | None = None
_request_metrics_cache: dict[str, _RequestMetrics] = {}

# ---------------------------------------------------------------------------
# Pydantic models for structured metric access
# ---------------------------------------------------------------------------


class TaskMetrics(BaseModel):
    """Task queue metrics container.

    Attributes:
        queue_depth: Current number of queued tasks.
        tasks_total: Total task completions.
        execution_seconds: Pure task execution time in seconds.
        queue_wait_seconds: Time from enqueue to execution start.
        rate_limit_rejections: Total rate-limit rejections.
        rate_limiter_active_tokens: Tokens tracked by rate limiter.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    queue_depth: Gauge
    tasks_total: Counter
    execution_seconds: Histogram
    queue_wait_seconds: Histogram
    rate_limit_rejections: Counter
    rate_limiter_active_tokens: Gauge


class ResourceMetrics(BaseModel):
    """Resource gauge metrics container.

    Attributes:
        active_sessions: Current active session count.
        active_models: Current registered model count.
        active_sampling_sessions: Current sampling session count.
        active_futures: Current future/request count.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    active_sessions: Gauge
    active_models: Gauge
    active_sampling_sessions: Gauge
    active_futures: Gauge


class _RequestMetrics(BaseModel):
    """HTTP request metrics container (internal)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    requests_total: Counter
    request_duration_seconds: Histogram


# ---------------------------------------------------------------------------
# A.  Request-level metrics  (FastAPI middleware)
# ---------------------------------------------------------------------------


def _get_request_metrics(deployment: str) -> _RequestMetrics:
    """Return (or create) per-deployment HTTP request metrics."""
    if deployment in _request_metrics_cache:
        return _request_metrics_cache[deployment]

    metrics = _RequestMetrics(
        requests_total=Counter(
            'twinkle_requests_total',
            description='Total HTTP requests.',
            tag_keys=('deployment', 'method', 'status'),
        ),
        request_duration_seconds=Histogram(
            'twinkle_request_duration_seconds',
            description='End-to-end HTTP request latency in seconds.',
            boundaries=_HISTOGRAM_BOUNDARIES,
            tag_keys=('deployment', 'method'),
        ),
    )
    _request_metrics_cache[deployment] = metrics
    return metrics


def create_metrics_middleware(deployment: str) -> Callable:
    """Return a FastAPI ``http`` middleware that records request metrics.

    Usage inside a ``build_*_app()`` function::

        from twinkle.server.utils.metrics import create_metrics_middleware
        metrics_mw = create_metrics_middleware("Model")
        app.middleware('http')(metrics_mw)

    Because FastAPI executes middleware in LIFO order, registering this
    **after** ``verify_token`` means it wraps the outermost layer and
    captures full end-to-end latency including auth.
    """

    async def metrics_middleware(request: Any, call_next: Callable) -> Any:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        status = str(response.status_code)
        method = request.scope['route'].path if 'route' in request.scope else request.url.path
        m = _get_request_metrics(deployment)
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


# ---------------------------------------------------------------------------
# B.  Task-queue metrics
# ---------------------------------------------------------------------------


def get_task_metrics(deployment: str) -> TaskMetrics:
    """Return (or create) per-deployment task-queue metrics.

    Returns a :class:`TaskMetrics` Pydantic model with:

    - ``queue_depth``                – Gauge
    - ``tasks_total``                – Counter
    - ``execution_seconds``          – Histogram
    - ``queue_wait_seconds``         – Histogram
    - ``rate_limit_rejections``      – Counter
    - ``rate_limiter_active_tokens`` – Gauge
    """
    if deployment in _task_metrics_cache:
        return _task_metrics_cache[deployment]

    metrics = TaskMetrics(
        queue_depth=Gauge(
            'twinkle_task_queue_depth',
            description='Current number of queued tasks.',
            tag_keys=('deployment', ),
        ),
        tasks_total=Counter(
            'twinkle_tasks_total',
            description='Total task completions.',
            tag_keys=('deployment', 'task_type', 'status'),
        ),
        execution_seconds=Histogram(
            'twinkle_task_execution_seconds',
            description='Pure task execution time in seconds.',
            boundaries=_HISTOGRAM_BOUNDARIES,
            tag_keys=('deployment', 'task_type'),
        ),
        queue_wait_seconds=Histogram(
            'twinkle_task_queue_wait_seconds',
            description='Time from enqueue to execution start in seconds.',
            boundaries=_HISTOGRAM_BOUNDARIES,
            tag_keys=('deployment', 'task_type'),
        ),
        rate_limit_rejections=Counter(
            'twinkle_rate_limit_rejections_total',
            description='Total rate-limit rejections.',
            tag_keys=('deployment', ),
        ),
        rate_limiter_active_tokens=Gauge(
            'twinkle_rate_limiter_active_tokens',
            description='Number of tokens tracked by the rate limiter.',
            tag_keys=('deployment', ),
        ),
    )
    _task_metrics_cache[deployment] = metrics
    return metrics


# ---------------------------------------------------------------------------
# D.  Resource gauges  (ServerState actor, updated every 15 s)
# ---------------------------------------------------------------------------


def get_resource_metrics() -> ResourceMetrics:
    """Return (or create) global resource gauge metrics.

    Returns a :class:`ResourceMetrics` Pydantic model with:

    - ``active_sessions``           – Gauge
    - ``active_models``             – Gauge
    - ``active_sampling_sessions``  – Gauge
    - ``active_futures``            – Gauge
    """
    global _resource_metrics_cache
    if _resource_metrics_cache is not None:
        return _resource_metrics_cache

    metrics = ResourceMetrics(
        active_sessions=Gauge(
            'twinkle_active_sessions',
            description='Current active session count.',
        ),
        active_models=Gauge(
            'twinkle_active_models',
            description='Current registered model count.',
        ),
        active_sampling_sessions=Gauge(
            'twinkle_active_sampling_sessions',
            description='Current sampling session count.',
        ),
        active_futures=Gauge(
            'twinkle_active_futures',
            description='Current future/request count.',
        ),
    )
    _resource_metrics_cache = metrics
    return metrics
