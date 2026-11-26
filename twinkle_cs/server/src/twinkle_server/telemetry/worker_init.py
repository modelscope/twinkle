"""Start telemetry in Ray workers, which do not inherit the driver's providers."""
from __future__ import annotations
from typing import TYPE_CHECKING

from twinkle_server.config.telemetry import TelemetryConfig
from twinkle.utils import get_logger
from .provider import init_telemetry, shutdown_telemetry

if TYPE_CHECKING:
    from .resource_metrics import ResourceMetricsCollector

logger = get_logger()
_node_metrics: ResourceMetricsCollector | None = None


def ensure_telemetry_initialized() -> None:
    """Start the process pipeline and its node and process gauges once."""
    global _node_metrics
    if _node_metrics is not None:
        return
    try:
        config = TelemetryConfig.from_env()
        if init_telemetry(config):
            from .resource_metrics import ResourceMetricsCollector

            _node_metrics = ResourceMetricsCollector(config)
            _node_metrics.maybe_start()
    except Exception as exc:
        logger.warning('Failed to initialize worker telemetry (%s)', type(exc).__name__)


def shutdown_telemetry_safely() -> None:
    """Flush buffered signals, close providers, and release this worker's node metrics lease."""
    global _node_metrics
    try:
        shutdown_telemetry()
        if _node_metrics is not None:
            _node_metrics.close()
            _node_metrics = None
    except Exception as exc:  # pragma: no cover - telemetry must not interrupt worker teardown
        logger.warning('Telemetry shutdown failed (%s)', type(exc).__name__)
