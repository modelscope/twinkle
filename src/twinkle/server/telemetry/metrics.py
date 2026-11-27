"""Process-local OpenTelemetry instruments shared by server components.

The OTel proxy meter binds instruments created before provider startup.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from twinkle.server.exceptions import ResourceReleaseError
from .provider import get_meter

METERED_TASK_TYPES = frozenset({
    'forward', 'forward_only', 'forward_backward', 'step', 'clip_grad_and_step', 'optim_step', 'sample',
})

try:
    from opentelemetry.metrics import Observation
except Exception:  # pragma: no cover - OTEL not installed; NoopMeter never invokes the callback below.
    Observation = None  # type: ignore[assignment, misc]

_RESOURCE_GAUGES: tuple[tuple[str, str, str], ...] = (
    ('active_sessions', 'twinkle.sessions.active', 'Number of active client sessions'),
    ('active_models', 'twinkle.models.active', 'Number of registered models'),
    ('active_sampling_sessions', 'twinkle.sampling_sessions.active', 'Number of active sampling sessions'),
    ('active_futures', 'twinkle.futures.active', 'Number of retained task results, including pending futures'),
)

class _AbsoluteUpDownCounter:
    """Record changes to an absolute value for each attribute set."""

    def __init__(self, instrument: Any) -> None:
        self._instrument = instrument
        self._last: dict[tuple, float] = {}

    def set(self, value: float, attributes: dict[str, str] | None = None) -> None:
        attrs = attributes or {}
        key = tuple(sorted(attrs.items()))
        delta = value - self._last.get(key, 0)
        if delta:
            self._instrument.add(delta, attributes=attrs)
        if value:
            self._last[key] = value
        else:
            self._last.pop(key, None)


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
        self.queue_depth = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.queue.depth',
            description='Current task queue depth',
        ))
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
        self.admission_rejections = meter.create_counter(
            'twinkle.admission.rejections.total',
            description='Business admission rejections counted where the decision was made',
        )
        self.tasks_total = meter.create_counter(
            'twinkle.tasks.total',
            description='Total task completions, partitioned by status',
        )
        self.rate_limiter_active_tokens = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.rate_limiter.active_tokens',
            description='Number of tokens currently tracked by the rate limiter',
        ))

        self.usage_requests = meter.create_counter(
            'twinkle.usage.requests.total', description='Admitted training compute and sampling requests')
        self.usage_input_tokens = meter.create_counter(
            'twinkle.usage.input_tokens.total', description='Input tokens of admitted compute requests')

        # === Managed training ===
        self.training_instances_active = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.training.instances.active', description='Active managed training instances by lifecycle state'))
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
        self.training_devices_allocated = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.training.devices.allocated', description='Logical devices owned by managed training instances'))
        self.training_placement_groups_active = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.training.placement_groups.active',
            description='Placement groups owned by managed training instances'))

        # === Redis state backend ===
        self.redis_health = _AbsoluteUpDownCounter(meter.create_up_down_counter(
            'twinkle.state.redis.health', description='Last observed Redis health (1 healthy, 0 unhealthy)'))
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

    def _make_gauge_callback(self, name: str):
        """Build the sync OTEL callback that reads ``_resource_cache[name]``."""

        def _callback(options):  # OTEL signature
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

    @classmethod
    def get(cls) -> MetricsRegistry:
        """Retrieve global MetricsRegistry singleton. Created on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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

    def __init__(self, metrics: MetricsRegistry):
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
            tags = dict(zip(('backend', 'mode', 'state', 'resident'), key, strict=False))
            self.metrics.training_instances_active.set(counts.get(key, 0), attributes=tags)
        for key in self._resource_keys | resources.keys():
            tags = dict(zip(('backend', 'mode', 'device_type'), key, strict=False))
            devices, groups = resources.get(key, (0, 0))
            self.metrics.training_devices_allocated.set(devices, attributes=tags)
            self.metrics.training_placement_groups_active.set(groups, attributes=tags)
        self._instance_keys = set(counts)
        self._resource_keys = set(resources)

    @staticmethod
    def quarantine_reason(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return 'timeout'
        if isinstance(exc, ResourceReleaseError):
            return exc.phase
        return 'other'
