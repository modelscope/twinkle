"""Twinkle Server metrics registry — low-invasiveness facade over OpenTelemetry metrics."""

from __future__ import annotations

from .provider import get_meter


class MetricsRegistry:
    """集中声明所有指标。业务代码通过 MetricsRegistry.get() 获取单例使用。

    当 telemetry 未初始化时，OTEL 返回 NoOp meter，所有记录操作自动静默。
    """

    _instance: MetricsRegistry | None = None

    def __init__(self) -> None:
        meter = get_meter("twinkle-server")

        # === HTTP 请求 ===
        self.requests_total = meter.create_counter(
            "twinkle.http.requests.total",
            description="Total HTTP requests received",
        )
        self.request_duration_seconds = meter.create_histogram(
            "twinkle.http.request.duration_seconds",
            description="HTTP request duration in seconds",
            unit="s",
        )

        # === 任务队列 ===
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

        # === 资源 ===
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
        """获取全局 MetricsRegistry 单例。首次调用时创建。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（用于测试或 telemetry 重新初始化）"""
        cls._instance = None
