# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource (CPU / Memory / GPU) observable gauges.

Registers OTEL ``observable_gauge`` instruments for CPU utilization, memory
usage (system + process), GPU utilization, and GPU memory. The data sources
(``psutil`` for CPU/memory, ``pynvml`` for GPU) are **optional telemetry
dependencies** — when either is missing, the corresponding gauges report no
data and the collector does not raise.

The collector is started by :func:`worker_init.ensure_telemetry_initialized`
in each worker. One process per node and telemetry destination reports node
metrics, while each worker reports its own RSS.
"""
from __future__ import annotations
import fcntl
import os
import tempfile
from collections.abc import Iterator
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Any

from .correlation import telemetry_node
from .provider import get_meter

try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False

try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

_NVML_INITIALIZED = False


def _nvml_handle_count() -> int:
    """Return GPU count, initializing pynvml lazily. ``0`` if unavailable."""
    global _NVML_INITIALIZED
    if not _PYNVML_AVAILABLE:
        return 0
    try:
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:
        return 0


class ResourceMetricsCollector:
    """Owns the observable-gauge callbacks for CPU/Mem/GPU."""

    def __init__(self) -> None:
        self._started = False
        self._node = telemetry_node()
        self._node_attrs = {'node': self._node}
        self._process_attrs = {**self._node_attrs, 'pid': os.getpid()}
        scope = (
            self._node,
            os.environ.get('TWINKLE_TELEMETRY_SERVICE', 'twinkle-server'),
            os.environ.get('TWINKLE_TELEMETRY_ENDPOINT', 'http://localhost:4317'),
        )
        self._lock_scope = sha256(repr(scope).encode()).hexdigest()[:16]
        self._node_leases = {}
        # Track which gauges were registered so callers (and tests) can
        # introspect what's exported in this process.
        self.registered_gauges: list[str] = []

    def maybe_start(self) -> None:
        """Register the available gauges; idempotent and never raises."""
        if self._started:
            return
        self._started = True
        try:
            meter = get_meter('twinkle.server.resource')
        except Exception:
            return

        # CPU + memory require psutil.
        if _PSUTIL_AVAILABLE:
            try:
                meter.create_observable_gauge(
                    'twinkle.system.cpu.utilization',
                    description='System CPU utilization (0..1)',
                    callbacks=[self._cpu_utilization_callback],
                )
                self.registered_gauges.append('twinkle.system.cpu.utilization')
                meter.create_observable_gauge(
                    'twinkle.system.memory.usage_bytes',
                    description='System memory used in bytes',
                    callbacks=[self._memory_usage_callback],
                )
                self.registered_gauges.append('twinkle.system.memory.usage_bytes')
                meter.create_observable_gauge(
                    'twinkle.process.memory.usage_bytes',
                    description='Resident-set memory of this process in bytes',
                    callbacks=[self._process_memory_callback],
                )
                self.registered_gauges.append('twinkle.process.memory.usage_bytes')
            except Exception:
                pass

        # GPU requires pynvml AND at least one GPU device — without either,
        # we silently skip GPU gauges.
        if _nvml_handle_count() > 0:
            try:
                meter.create_observable_gauge(
                    'twinkle.gpu.utilization',
                    description='Per-GPU utilization (0..1)',
                    callbacks=[self._gpu_utilization_callback],
                )
                self.registered_gauges.append('twinkle.gpu.utilization')
                meter.create_observable_gauge(
                    'twinkle.gpu.memory.usage_bytes',
                    description='Per-GPU memory used in bytes',
                    callbacks=[self._gpu_memory_callback],
                )
                self.registered_gauges.append('twinkle.gpu.memory.usage_bytes')
            except Exception:
                pass

    def _owns_node_metrics(self, group: str) -> bool:
        lease = self._node_leases.get(group)
        try:
            if lease is None:
                path = Path(tempfile.gettempdir()) / f'twinkle-{os.getuid()}-{self._lock_scope}-{group}.lock'
                lease = path.open('a')
                self._node_leases[group] = lease
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def close(self) -> None:
        for lease in self._node_leases.values():
            lease.close()
        self._node_leases.clear()

    # ----- callbacks ----------------------------------------------------- #

    def _cpu_utilization_callback(self, _options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE or not self._owns_node_metrics('system'):
            return iter(())
        try:
            value = float(psutil.cpu_percent(interval=None)) / 100.0
            return iter([Observation(value, self._node_attrs)])
        except Exception:
            return iter(())

    def _memory_usage_callback(self, _options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE or not self._owns_node_metrics('system'):
            return iter(())
        try:
            return iter([Observation(int(psutil.virtual_memory().used), self._node_attrs)])
        except Exception:
            return iter(())

    def _process_memory_callback(self, _options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE:
            return iter(())
        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
            return iter([Observation(int(rss), self._process_attrs)])
        except Exception:
            return iter(())

    @cached_property
    def _devices(self) -> list[tuple[Any, dict[str, str | int]]]:
        devices = []
        for index in range(_nvml_handle_count()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                attrs = {
                    **self._node_attrs,
                    'gpu_index': index,
                    'gpu_uuid': uuid.decode() if isinstance(uuid, bytes) else uuid,
                }
                devices.append((handle, attrs))
            except Exception:
                continue
        return devices

    def _gpu_utilization_callback(self, _options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation

        if self._owns_node_metrics('gpu'):
            for handle, attrs in self._devices:
                try:
                    yield Observation(float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu) / 100.0, attrs)
                except Exception:
                    continue

    def _gpu_memory_callback(self, _options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation

        if self._owns_node_metrics('gpu'):
            for handle, attrs in self._devices:
                try:
                    yield Observation(int(pynvml.nvmlDeviceGetMemoryInfo(handle).used), attrs)
                except Exception:
                    continue


_GLOBAL_COLLECTOR: ResourceMetricsCollector | None = None


def get_collector() -> ResourceMetricsCollector:
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        _GLOBAL_COLLECTOR = ResourceMetricsCollector()
    return _GLOBAL_COLLECTOR


def reset_collector_for_tests() -> None:
    """Clear the module-global collector. Test-only helper."""
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is not None:
        _GLOBAL_COLLECTOR.close()
    _GLOBAL_COLLECTOR = None
