"""Twinkle Server metrics registry — low-invasiveness facade over OpenTelemetry metrics."""

from __future__ import annotations

from .provider import get_meter


class MetricsRegistry:
    """Centrally declares all metrics. Business code retrieves singleton via MetricsRegistry.get().

    When telemetry is not initialized, OTEL returns a NoOp meter and all recording operations are silently no-op.
    """

    _instance: MetricsRegistry | None = None

    def __init__(self) -> None:
        meter = get_meter("twinkle-server")

        # === HTTP Requests ===
        self.requests_total = meter.create_counter(
            "twinkle.http.requests.total",
            description="Total HTTP requests received",
        )
        self.request_duration_seconds = meter.create_histogram(
            "twinkle.http.request.duration_seconds",
            description="HTTP request duration in seconds",
            unit="s",
        )

        # === Task Queue ===
        self.queue_depth = meter.create_up_down_counter(
            "twinkle.queue.depth",
            description="Current task queue depth",
        )
        self.task_execution_seconds = meter.create_histogram(
            "twinkle.task.execution_seconds",
            description="Task execution duration in seconds",
            unit="s",
        )
        self.task_wait_seconds = meter.create_histogram(
            "twinkle.task.wait_seconds",
            description="Task wait time in queue before execution",
            unit="s",
        )
        self.rate_limit_rejections = meter.create_counter(
            "twinkle.rate_limit.rejections.total",
            description="Total requests rejected by rate limiter",
        )
        self.tasks_total = meter.create_counter(
            "twinkle.tasks.total",
            description="Total task completions, partitioned by status",
        )
        self.rate_limiter_active_tokens = meter.create_up_down_counter(
            "twinkle.rate_limiter.active_tokens",
            description="Number of tokens currently tracked by the rate limiter",
        )

        # === Resources ===
        self.active_sessions = meter.create_up_down_counter(
            "twinkle.sessions.active",
            description="Number of active client sessions",
        )
        self.active_models = meter.create_up_down_counter(
            "twinkle.models.active",
            description="Number of registered models",
        )
        self.active_sampling_sessions = meter.create_up_down_counter(
            "twinkle.sampling_sessions.active",
            description="Number of active sampling sessions",
        )
        self.active_futures = meter.create_up_down_counter(
            "twinkle.futures.active",
            description="Number of pending futures/tasks",
        )

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
