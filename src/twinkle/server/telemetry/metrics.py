"""Twinkle Server metrics registry — low-invasiveness facade over OpenTelemetry metrics."""

from __future__ import annotations

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

    def _make_gauge_callback(self, name: str):
        """Build the sync OTEL callback that reads ``_resource_cache[name]``."""

        def _callback(options):  # noqa: ARG001 -- OTEL signature
            return [Observation(self._resource_cache.get(name, 0))]

        return _callback

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
