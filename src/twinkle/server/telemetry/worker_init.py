"""Telemetry initialization for Ray worker processes.

Ray Serve deployments run in separate worker processes that do not inherit
the driver process's OTEL global state. This module provides a function
to re-initialize telemetry in each worker process using environment variables
set by the launcher.
"""
from __future__ import annotations

import os

from twinkle.utils import get_logger

logger = get_logger()

_worker_initialized = False


def ensure_telemetry_initialized() -> None:
    """Initialize telemetry in the current worker process if not already done.

    Reads configuration from environment variables set by the launcher process.
    Safe to call multiple times - only initializes once per process.
    """
    global _worker_initialized
    if _worker_initialized:
        return

    _worker_initialized = True

    telemetry_enabled = os.environ.get('TWINKLE_TELEMETRY_ENABLED') == '1'

    if not telemetry_enabled:
        # Even with telemetry disabled, register the resource collector so
        # graceful-degradation behavior matches the enabled path.
        _start_resource_collector()
        return

    try:
        from twinkle.server.telemetry import TelemetryConfig, init_telemetry
        from twinkle.server.telemetry.metrics import MetricsRegistry

        config = TelemetryConfig(
            enabled=True,
            debug=os.environ.get('TWINKLE_TELEMETRY_DEBUG', '0') == '1',
            service_name=os.environ.get('TWINKLE_TELEMETRY_SERVICE', 'twinkle-server'),
            otlp_endpoint=os.environ.get('TWINKLE_TELEMETRY_ENDPOINT', 'http://localhost:4317'),
            export_interval_ms=int(os.environ.get('TWINKLE_TELEMETRY_INTERVAL', '30000')),
        )
        init_telemetry(config)
        # Reset MetricsRegistry singleton so it picks up the real MeterProvider
        MetricsRegistry.reset()
        logger.info(f'Worker telemetry initialized (service={config.service_name}, debug={config.debug})')
    except Exception as e:
        logger.warning(f'Failed to initialize worker telemetry: {e}')

    _start_resource_collector()


def _start_resource_collector() -> None:
    """Start the resource (CPU / Memory / GPU) metrics collector.

    Safe to call even when telemetry init was skipped or failed — the
    collector picks up the NoOp meter and silently records no observations
    the collector picks up the NoOp meter and silently records nothing.
    """
    try:
        from twinkle.server.telemetry import resource_metrics

        resource_metrics.get_collector().maybe_start()
    except Exception as e:
        logger.debug(f'Resource metrics collector start failed: {e}')


def flush_telemetry_safely() -> None:
    """Flush + shut down telemetry, swallowing any error.

    Called from the launcher driver after ``serve.shutdown()`` and from each
    worker deployment's FastAPI lifespan shutdown so buffered OTLP batches
    (traces / metrics / logs) flush on graceful termination. A
    telemetry-shutdown failure MUST NOT mask the user-facing shutdown path,
    so every error here is swallowed (Requirement 21.3).
    """
    try:
        from twinkle.server.telemetry import shutdown_telemetry
        shutdown_telemetry()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f'Telemetry shutdown failed: {e}')
