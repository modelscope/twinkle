# Copyright (c) ModelScope Contributors. All rights reserved.
"""Collection of telemetry / persistence env vars for propagation to Ray workers.

Extracted from the former single-file ``launcher.py`` (TIER 3 same-named-package
decomposition). No logic change. These vars are read inside each Ray Serve
worker process — telemetry by ``ensure_telemetry_initialized()`` and persistence
by ``PersistenceConfig.from_env()`` — so the chosen backend / telemetry config
is independent of deployment startup order.
"""
from __future__ import annotations

import os

# Telemetry env var keys that need to be propagated to Ray worker processes.
TELEMETRY_ENV_KEYS: tuple[str, ...] = (
    'TWINKLE_TELEMETRY_ENABLED',
    'TWINKLE_TELEMETRY_DEBUG',
    'TWINKLE_TELEMETRY_SERVICE',
    'TWINKLE_TELEMETRY_ENDPOINT',
    'TWINKLE_TELEMETRY_INTERVAL',
    'TWINKLE_MODEL_ID_ALIASES',
)

# NCCL-safe env var keys: controls fault tolerance behavior in distributed
# training (safe_loss / @nccl_safe). Must reach model worker actors.
NCCL_SAFE_ENV_KEYS: tuple[str, ...] = ('TWINKLE_FAIL_FAST', )


def build_telemetry_env_vars() -> dict[str, str]:
    """Collect telemetry env vars from ``os.environ`` for worker propagation."""
    return {k: os.environ[k] for k in TELEMETRY_ENV_KEYS if k in os.environ}


def build_persistence_env_vars() -> dict[str, str]:
    """Collect persistence env vars from ``os.environ`` for worker propagation."""
    from twinkle.server.config.persistence import PERSISTENCE_ENV_KEYS
    return {k: os.environ[k] for k in PERSISTENCE_ENV_KEYS if k in os.environ}


def build_nccl_safe_env_vars() -> dict[str, str]:
    """Collect NCCL-safe env vars from ``os.environ`` for worker propagation."""
    return {k: os.environ[k] for k in NCCL_SAFE_ENV_KEYS if k in os.environ}


def build_propagated_env_vars() -> dict[str, str]:
    """Aggregate all env vars that must reach Ray worker processes."""
    merged: dict[str, str] = {}
    merged.update(build_telemetry_env_vars())
    merged.update(build_persistence_env_vars())
    merged.update(build_nccl_safe_env_vars())
    return merged
