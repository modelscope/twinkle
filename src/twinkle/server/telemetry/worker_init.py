"""Telemetry initialization for Ray worker processes.

Ray Serve deployments run in separate worker processes that do not inherit
the driver process's OTEL global state. This module provides a function
to re-initialize telemetry in each worker process using environment variables
set by the launcher.
"""
from __future__ import annotations
import os

import json

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

    if os.environ.get('TWINKLE_TELEMETRY_ENABLED') != '1':
        return

    try:
        from twinkle.server.config.telemetry import TelemetryConfig
        from twinkle.server.telemetry.provider import init_telemetry
        from twinkle.server.telemetry.resource_metrics import get_collector

        config = TelemetryConfig(
            enabled=True,
            debug=os.environ.get('TWINKLE_TELEMETRY_DEBUG', '0') == '1',
            service_name=os.environ.get('TWINKLE_TELEMETRY_SERVICE', 'twinkle-server'),
            otlp_endpoint=os.environ.get('TWINKLE_TELEMETRY_ENDPOINT', 'http://localhost:4317'),
            export_interval_ms=int(os.environ.get('TWINKLE_TELEMETRY_INTERVAL', '30000')),
            resource_attributes=json.loads(os.environ.get('TWINKLE_TELEMETRY_RESOURCE_ATTRIBUTES', '{}')),
        )
        if init_telemetry(config):
            get_collector().maybe_start()
            _worker_initialized = True
    except Exception as exc:
        logger.warning('Failed to initialize worker telemetry (%s)', type(exc).__name__)


def flush_telemetry_safely() -> None:
    """Flush + shut down telemetry, swallowing any error.

    Called from the launcher driver after ``serve.shutdown()`` and from each
    worker deployment's FastAPI lifespan shutdown so buffered OTLP batches
    (traces / metrics / logs) flush on graceful termination. A
    telemetry-shutdown failure MUST NOT mask the user-facing shutdown path,
    so every error here is swallowed.
    """
    try:
        from twinkle.server.telemetry.provider import shutdown_telemetry
        shutdown_telemetry()
        if _worker_initialized:
            from twinkle.server.telemetry.resource_metrics import get_collector
            get_collector().close()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning('Telemetry shutdown failed (%s)', type(e).__name__)
