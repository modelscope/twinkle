"""Twinkle Server metrics registry — low-invasiveness facade over OpenTelemetry metrics.

Besides the :class:`MetricsRegistry` that declares the raw OTEL instruments, this module
holds the legacy-API adapter classes (``_Counter`` / ``_Histogram`` / ``_Gauge``), the
structured containers (:class:`TaskMetrics` / ``_RequestMetrics``) and ``get_task_metrics``.
Keeping the metric types and the registry in one file (with the HTTP middleware factory in
``http_middleware.py``) avoids a ``metrics <-> adapters`` import cycle.

Per-deployment adapters are cached on the *registry instance*
(``_task_metrics`` / ``_request_metrics``), not at module level. That is load-bearing: the
adapters hold bound instrument objects, and ``reset()`` swaps the singleton precisely in
order to rebind them to a real MeterProvider (``worker_init.ensure_telemetry_initialized``
does ``init_telemetry()`` -> ``reset()``). A module-level cache survived that swap, so
anything that called ``get_task_metrics`` before ``init_telemetry`` cached NoOp instruments
for the life of the process.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import Any

from .provider import get_meter

try:
    from opentelemetry.metrics import Observation
except Exception:  # pragma: no cover - OTEL not installed; NoopMeter never invokes the callback below.
    Observation = None  # type: ignore[assignment, misc]

_RESOURCE_GAUGES: tuple[tuple[str, str, str], ...] = (
    ('active_sessions', 'twinkle.sessions.active', 'Number of active client sessions'),
    ('active_models', 'twinkle.models.active', 'Number of registered models'),
    ('active_sampling_sessions', 'twinkle.sampling_sessions.active', 'Number of active sampling sessions'),
    ('active_futures', 'twinkle.futures.active', 'Number of pending futures/tasks'),
)

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


class MetricsRegistry:
    """Centrally declares all metrics. Business code retrieves singleton via MetricsRegistry.get().

    Resource counters (sessions/models/sampling_sessions/futures) are
    ObservableGauges fed from a cache dict. Whichever ``ServerState`` instance
    holds the cleanup leader lease is responsible for pushing fresh values via
    :meth:`set_resource_count`; non-leader instances stay silent so the four
    Ray Serve worker processes do not multiply the reported counts.

    When telemetry is not initialized, OTEL returns a NoOp meter and all
    recording operations are silently no-op.
    """

    _instance: MetricsRegistry | None = None

    def __init__(self) -> None:
        meter = get_meter('twinkle-server')

        # === HTTP Requests ===
        self.requests_total = meter.create_counter(
            'twinkle.http.requests.total',
            description='Total HTTP requests received',
        )
        self.request_duration_seconds = meter.create_histogram(
            'twinkle.http.request.duration_seconds',
            description='HTTP request duration in seconds',
            unit='s',
        )

        # === Task Queue ===
        self.queue_depth = meter.create_up_down_counter(
            'twinkle.queue.depth',
            description='Current task queue depth',
        )
        self.task_execution_seconds = meter.create_histogram(
            'twinkle.task.execution_seconds',
            description='Task execution duration in seconds',
            unit='s',
        )
        self.task_wait_seconds = meter.create_histogram(
            'twinkle.task.wait_seconds',
            description='Task wait time in queue before execution',
            unit='s',
        )
        self.rate_limit_rejections = meter.create_counter(
            'twinkle.rate_limit.rejections.total',
            description='Total requests rejected by rate limiter',
        )
        self.tasks_total = meter.create_counter(
            'twinkle.tasks.total',
            description='Total task completions, partitioned by status',
        )
        self.rate_limiter_active_tokens = meter.create_up_down_counter(
            'twinkle.rate_limiter.active_tokens',
            description='Number of tokens currently tracked by the rate limiter',
        )

        # === Resources (ObservableGauge backed by the push cache) ===
        # OTEL holds its own references to the gauges via the meter, so we
        # only need to keep the cache the callbacks read from.
        self._resource_cache: dict[str, int] = {name: 0 for name, _, _ in _RESOURCE_GAUGES}
        for attr_name, otel_name, description in _RESOURCE_GAUGES:
            meter.create_observable_gauge(
                otel_name,
                callbacks=[self._make_gauge_callback(attr_name)],
                description=description,
            )

        # Per-deployment adapter caches held on the instance so ``reset()`` (which
        # swaps this singleton to rebind instruments) invalidates them.
        self._task_metrics: dict[str, TaskMetrics] = {}
        self._request_metrics: dict[str, _RequestMetrics] = {}

    def _make_gauge_callback(self, name: str):
        """Build the sync OTEL callback that reads ``_resource_cache[name]``."""

        def _callback(options):  # noqa: ARG001 -- OTEL signature
            return [Observation(self._resource_cache.get(name, 0))]

        return _callback

    # ----- Per-deployment adapter accessors -----

    def task_metrics(self, deployment: str) -> TaskMetrics:
        """Return (or build) the per-deployment task-queue metric adapters."""
        cached = self._task_metrics.get(deployment)
        if cached is None:
            cached = self._task_metrics[deployment] = TaskMetrics(
                queue_depth=_Gauge(self.queue_depth),
                tasks_total=_Counter(self.tasks_total),
                execution_seconds=_Histogram(self.task_execution_seconds),
                queue_wait_seconds=_Histogram(self.task_wait_seconds),
                rate_limit_rejections=_Counter(self.rate_limit_rejections),
                rate_limiter_active_tokens=_Gauge(self.rate_limiter_active_tokens),
            )
        return cached

    def request_metrics(self, deployment: str) -> _RequestMetrics:
        """Return (or build) the per-deployment HTTP request metric adapters."""
        cached = self._request_metrics.get(deployment)
        if cached is None:
            cached = self._request_metrics[deployment] = _RequestMetrics(
                requests_total=_Counter(self.requests_total),
                request_duration_seconds=_Histogram(self.request_duration_seconds),
            )
        return cached

    # ----- Push API for the cleanup leader -----

    def set_resource_count(self, name: str, value: int) -> None:
        """Update the cached value the matching ObservableGauge will report next."""
        if name in self._resource_cache:
            self._resource_cache[name] = int(value)

    def clear_resource_counts(self) -> None:
        """Reset every resource gauge to 0. Called when a worker loses leadership."""
        for name in self._resource_cache:
            self._resource_cache[name] = 0

    def get_resource_count(self, name: str) -> int:
        """Return the most recently pushed value for ``name`` (0 if never set)."""
        return self._resource_cache.get(name, 0)

    @classmethod
    def get(cls) -> MetricsRegistry:
        """Retrieve global MetricsRegistry singleton. Created on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing or telemetry re-initialization)."""
        cls._instance = None


def get_task_metrics(deployment: str) -> TaskMetrics:
    """Return the per-deployment task-queue metric adapters.

    Signature unchanged (``_init_task_queue`` needs no edit); the adapters are now
    cached on the ``MetricsRegistry`` instance so ``reset()`` invalidates them.
    """
    return MetricsRegistry.get().task_metrics(deployment)


def get_request_metrics(deployment: str) -> _RequestMetrics:
    """Return the per-deployment HTTP request metric adapters."""
    return MetricsRegistry.get().request_metrics(deployment)
