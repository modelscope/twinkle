# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource (CPU / Memory / GPU) observable gauges.

Registers OTEL ``observable_gauge`` instruments for CPU utilization, memory
usage (system + process), GPU utilization, and GPU memory. The data sources
(``psutil`` for CPU/memory, ``pynvml`` for GPU) are **optional telemetry
dependencies** — when either is missing, the corresponding gauges report no
data and the collector does not raise.

The collector is started by :func:`worker_init.ensure_telemetry_initialized`
in each Ray Serve worker process so per-replica resource usage shows up in
Mimir / Grafana.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

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

    # ----- callbacks ----------------------------------------------------- #

    @staticmethod
    def _cpu_utilization_callback(_options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE:
            return iter(())
        try:
            value = float(psutil.cpu_percent(interval=None)) / 100.0
            return iter([Observation(value)])
        except Exception:
            return iter(())

    @staticmethod
    def _memory_usage_callback(_options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE:
            return iter(())
        try:
            return iter([Observation(int(psutil.virtual_memory().used))])
        except Exception:
            return iter(())

    @staticmethod
    def _process_memory_callback(_options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        if not _PSUTIL_AVAILABLE:
            return iter(())
        try:
            rss = psutil.Process(os.getpid()).memory_info().rss
            return iter([Observation(int(rss))])
        except Exception:
            return iter(())

    @staticmethod
    def _gpu_utilization_callback(_options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        count = _nvml_handle_count()
        out: list[Any] = []
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                out.append(Observation(float(util.gpu) / 100.0, {'gpu_index': i}))
            except Exception:
                continue
        return iter(out)

    @staticmethod
    def _gpu_memory_callback(_options: Any) -> Iterator[Any]:
        from opentelemetry.metrics import Observation  # type: ignore

        count = _nvml_handle_count()
        out: list[Any] = []
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                out.append(Observation(int(mem.used), {'gpu_index': i}))
            except Exception:
                continue
        return iter(out)


_GLOBAL_COLLECTOR: ResourceMetricsCollector | None = None


def get_collector() -> ResourceMetricsCollector:
    global _GLOBAL_COLLECTOR
    if _GLOBAL_COLLECTOR is None:
        _GLOBAL_COLLECTOR = ResourceMetricsCollector()
    return _GLOBAL_COLLECTOR


def reset_collector_for_tests() -> None:
    """Clear the module-global collector. Test-only helper."""
    global _GLOBAL_COLLECTOR
    _GLOBAL_COLLECTOR = None
