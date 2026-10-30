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

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from typing import Any

from twinkle.server.exceptions import ResourceReleaseError
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


class TrainingMetrics(BaseModel):
    """Low-cardinality metrics for the managed training lifecycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instances_active: _Gauge
    startup_seconds: _Histogram
    shutdown_seconds: _Histogram
    allocation_seconds: _Histogram
    removal_seconds: _Histogram
    quarantined_total: _Counter
    sweep_total: _Counter
    sweep_seconds: _Histogram
    devices_allocated: _Gauge
    placement_groups_active: _Gauge


class StateBackendMetrics(BaseModel):
    """Metrics for persistent state backend availability and contention."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    redis_health: _Gauge
    update_atomic_conflicts: _Counter
    update_atomic_retries: _Histogram


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

        # === Managed training ===
        self.training_instances_active = meter.create_up_down_counter(
            'twinkle.training.instances.active', description='Active managed training instances by lifecycle state')
        self.training_startup_seconds = meter.create_histogram(
            'twinkle.training.instance.startup.duration_seconds',
            description='Training runtime startup duration',
            unit='s')
        self.training_shutdown_seconds = meter.create_histogram(
            'twinkle.training.instance.shutdown.duration_seconds',
            description='Training runtime shutdown duration',
            unit='s')
        self.training_allocation_seconds = meter.create_histogram(
            'twinkle.training.placement_group.allocation.duration_seconds',
            description='Placement group allocation duration',
            unit='s')
        self.training_removal_seconds = meter.create_histogram(
            'twinkle.training.placement_group.removal.duration_seconds',
            description='Placement group removal duration',
            unit='s')
        self.training_quarantined_total = meter.create_counter(
            'twinkle.training.quarantined.total',
            description='Transitions to quarantine during cleanup; retries in quarantine are excluded')
        self.training_sweep_total = meter.create_counter(
            'twinkle.training.sweep.total', description='Training lifecycle sweep runs by status')
        self.training_sweep_seconds = meter.create_histogram(
            'twinkle.training.sweep.duration_seconds', description='Training lifecycle sweep duration', unit='s')
        self.training_devices_allocated = meter.create_up_down_counter(
            'twinkle.training.devices.allocated', description='Logical devices owned by managed training instances')
        self.training_placement_groups_active = meter.create_up_down_counter(
            'twinkle.training.placement_groups.active',
            description='Placement groups owned by managed training instances')

        # === Redis state backend ===
        self.redis_health = meter.create_up_down_counter(
            'twinkle.state.redis.health', description='Last observed Redis health (1 healthy, 0 unhealthy)')
        self.redis_update_atomic_conflicts = meter.create_counter(
            'twinkle.state.redis.update_atomic.conflicts.total', description='Redis WATCH conflicts')
        self.redis_update_atomic_retries = meter.create_histogram(
            'twinkle.state.redis.update_atomic.retries', description='Retries used by Redis atomic updates')

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
        self._training_metrics: TrainingMetrics | None = None
        self._state_backend_metrics: StateBackendMetrics | None = None

    def _make_gauge_callback(self, name: str):
        """Build the sync OTEL callback that reads ``_resource_cache[name]``."""

        def _callback(options):  # OTEL signature
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

    def training_metrics(self) -> TrainingMetrics:
        """Return process-local managed training metric adapters."""
        if self._training_metrics is None:
            self._training_metrics = TrainingMetrics(
                instances_active=_Gauge(self.training_instances_active),
                startup_seconds=_Histogram(self.training_startup_seconds),
                shutdown_seconds=_Histogram(self.training_shutdown_seconds),
                allocation_seconds=_Histogram(self.training_allocation_seconds),
                removal_seconds=_Histogram(self.training_removal_seconds),
                quarantined_total=_Counter(self.training_quarantined_total),
                sweep_total=_Counter(self.training_sweep_total),
                sweep_seconds=_Histogram(self.training_sweep_seconds),
                devices_allocated=_Gauge(self.training_devices_allocated),
                placement_groups_active=_Gauge(self.training_placement_groups_active),
            )
        return self._training_metrics

    def state_backend_metrics(self) -> StateBackendMetrics:
        """Return process-local state backend metric adapters."""
        if self._state_backend_metrics is None:
            self._state_backend_metrics = StateBackendMetrics(
                redis_health=_Gauge(self.redis_health),
                update_atomic_conflicts=_Counter(self.redis_update_atomic_conflicts),
                update_atomic_retries=_Histogram(self.redis_update_atomic_retries),
            )
        return self._state_backend_metrics

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


def get_training_metrics() -> TrainingMetrics:
    """Return managed training metric adapters."""
    return MetricsRegistry.get().training_metrics()


def get_state_backend_metrics() -> StateBackendMetrics:
    """Return persistent state backend metric adapters."""
    return MetricsRegistry.get().state_backend_metrics()


@dataclass(frozen=True)
class TrainingInstanceSnapshot:
    """Known resource ownership, including draining/quarantined instances."""

    backend: str
    mode: str
    state: str
    resident: str
    device_type: str
    device_count: int
    placement_group_count: int


class TrainingInstanceMetrics:
    """Aggregate snapshots and clear retired label sets; never own scheduling state."""

    def __init__(self, metrics: TrainingMetrics):
        self.metrics = metrics
        self._instance_keys = set()
        self._resource_keys = set()

    def report(self, snapshots):
        counts, resources = {}, {}
        for snapshot in snapshots:
            key = snapshot.backend, snapshot.mode, snapshot.state, snapshot.resident
            counts[key] = counts.get(key, 0) + 1
            resource_key = snapshot.backend, snapshot.mode, snapshot.device_type
            devices, groups = resources.get(resource_key, (0, 0))
            resources[resource_key] = devices + snapshot.device_count, groups + snapshot.placement_group_count
        for key in self._instance_keys | counts.keys():
            tags = dict(zip(('backend', 'mode', 'state', 'resident'), key))
            self.metrics.instances_active.set(counts.get(key, 0), tags=tags)
        for key in self._resource_keys | resources.keys():
            tags = dict(zip(('backend', 'mode', 'device_type'), key))
            devices, groups = resources.get(key, (0, 0))
            self.metrics.devices_allocated.set(devices, tags=tags)
            self.metrics.placement_groups_active.set(groups, tags=tags)
        self._instance_keys = set(counts)
        self._resource_keys = set(resources)

    @staticmethod
    def quarantine_reason(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return 'timeout'
        if isinstance(exc, ResourceReleaseError):
            return exc.phase
        return 'other'
