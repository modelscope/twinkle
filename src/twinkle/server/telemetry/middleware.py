# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Central metrics module for Twinkle server observability.

This module is a **back-compat keyword shim plus a real ``_Gauge`` adapter** over
the OpenTelemetry instruments declared in
:class:`twinkle.server.telemetry.metrics.MetricsRegistry`. ``_Counter`` and
``_Histogram`` are thin pass-throughs whose only role is to accept the legacy
Ray-style ``tags=`` keyword and forward it as OTEL's ``attributes=``; ``_Gauge``
does real work — it translates the legacy ``set(value)`` API onto OTEL's
delta-based UpDownCounter by tracking the last reported value per attribute set.
Routing every measurement through OTEL while preserving the legacy call API
means existing call sites do not need to change.

Public entry-points (unchanged signatures):

* ``create_metrics_middleware(deployment)`` – FastAPI HTTP middleware
* ``get_task_metrics(deployment)``           – task-queue / rate-limit gauges
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pydantic import BaseModel, ConfigDict
from typing import Any

from twinkle.server.telemetry import MetricsRegistry
from twinkle.utils.logger import get_logger

logger = get_logger()

# Per-process caches; each Ray Serve worker holds its own instance.
_task_metrics_cache: dict[str, TaskMetrics] = {}
_request_metrics_cache: dict[str, _RequestMetrics] = {}

# ---------------------------------------------------------------------------
# Adapter classes – wrap OTEL instruments to expose the legacy Ray-style API
# (``.inc(tags=...)`` / ``.set(value, tags=...)`` / ``.observe(value, tags=...)``)
# while delegating all measurements to OpenTelemetry.
# ---------------------------------------------------------------------------


class _Counter:
    """Adapter mapping ``.inc(value, tags=...)`` to ``otel_counter.add()``."""

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument

    def inc(self, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        self._instrument.add(value, attributes=tags or {})


class _Histogram:
    """Adapter mapping ``.observe(value, tags=...)`` to ``otel_histogram.record()``."""

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument

    def observe(self, value: float, tags: dict[str, str] | None = None) -> None:
        self._instrument.record(value, attributes=tags or {})


class _Gauge:
    """Adapter mapping ``.set(value, tags=...)`` onto an OTEL UpDownCounter.

    OpenTelemetry up/down counters take *deltas*, not absolute values, so we
    track the last reported value per attribute combination and emit the
    incremental change. State is held per adapter instance (= per deployment),
    keyed by the frozen attribute tuple.
    """

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument
        self._last: dict[tuple, float] = {}

    def set(self, value: float, tags: dict[str, str] | None = None) -> None:
        attrs = tags or {}
        key = tuple(sorted(attrs.items()))
        last = self._last.get(key, 0.0)
        delta = value - last
        if delta != 0:
            self._instrument.add(delta, attributes=attrs)
        self._last[key] = value


# ---------------------------------------------------------------------------
# Pydantic containers for structured metric access
# ---------------------------------------------------------------------------


class TaskMetrics(BaseModel):
    """Task queue metrics container.

    Attributes:
        queue_depth: Current number of queued tasks (gauge).
        tasks_total: Total task completions (counter).
        execution_seconds: Pure task execution time in seconds (histogram).
        queue_wait_seconds: Time from enqueue to execution start (histogram).
        rate_limit_rejections: Total rate-limit rejections (counter).
        rate_limiter_active_tokens: Tokens tracked by rate limiter (gauge).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    queue_depth: _Gauge
    tasks_total: _Counter
    execution_seconds: _Histogram
    queue_wait_seconds: _Histogram
    rate_limit_rejections: _Counter
    rate_limiter_active_tokens: _Gauge


class _RequestMetrics(BaseModel):
    """HTTP request metrics container (internal)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    requests_total: _Counter
    request_duration_seconds: _Histogram


# ---------------------------------------------------------------------------
# A.  Request-level metrics  (FastAPI middleware)
# ---------------------------------------------------------------------------


def _get_request_metrics(deployment: str) -> _RequestMetrics:
    """Return (or create) per-deployment HTTP request metric adapters."""
    if deployment in _request_metrics_cache:
        return _request_metrics_cache[deployment]

    reg = MetricsRegistry.get()
    metrics = _RequestMetrics(
        requests_total=_Counter(reg.requests_total),
        request_duration_seconds=_Histogram(reg.request_duration_seconds),
    )
    _request_metrics_cache[deployment] = metrics
    return metrics


def create_metrics_middleware(deployment: str) -> Callable:
    """Return a FastAPI ``http`` middleware that records request metrics.

    Usage inside a ``build_*_app()`` function::

        from twinkle.server.telemetry.middleware import create_metrics_middleware
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
    """Return (or create) per-deployment task-queue metric adapters.

    Returns a :class:`TaskMetrics` container of adapter objects; the
    adapters delegate every measurement to the OTEL instruments held by
    :class:`twinkle.server.telemetry.metrics.MetricsRegistry`. A separate
    adapter instance is cached per deployment so that gauge-state tracking
    (last value per attribute set) stays isolated.
    """
    if deployment in _task_metrics_cache:
        return _task_metrics_cache[deployment]

    reg = MetricsRegistry.get()
    metrics = TaskMetrics(
        queue_depth=_Gauge(reg.queue_depth),
        tasks_total=_Counter(reg.tasks_total),
        execution_seconds=_Histogram(reg.task_execution_seconds),
        queue_wait_seconds=_Histogram(reg.task_wait_seconds),
        rate_limit_rejections=_Counter(reg.rate_limit_rejections),
        rate_limiter_active_tokens=_Gauge(reg.rate_limiter_active_tokens),
    )
    _task_metrics_cache[deployment] = metrics
    return metrics
