# Copyright (c) ModelScope Contributors. All rights reserved.
"""Environment variables propagated from the launcher to Ray workers."""

import os

from .telemetry import TELEMETRY_WORKER_ENV_KEYS

SHARED_WORKER_ENV_KEYS: tuple[str, ...] = (
    "TWINKLE_STATE_RAY_NAMESPACE",
    *TELEMETRY_WORKER_ENV_KEYS,
    "TWINKLE_TELEMETRY_USER_ID_KEY",
    "TWINKLE_MODEL_ID_ALIASES",
    "TWINKLE_TRUST_REMOTE_CODE",
    # Must reach every backend worker (RayRuntime.deployment_options), not just the
    # deployment replica: run_worker_setup_hook reads it to register startup plugins.
    "TWINKLE_WORKER_SETUP_HOOK",
    # Server-side perf instruments (checkpoint_manager save_training/save_sampler disk-write
    # latency) gate on this in the BACKEND worker's os.environ, so it must propagate there too;
    # without it the instruments stay silent even when the client sets TWINKLE_PERF=1.
    "TWINKLE_PERF",
)

PERSISTENCE_ENV_KEYS: tuple[str, ...] = (
    "TWINKLE_PERSISTENCE_MODE",
    "TWINKLE_PERSISTENCE_REDIS_URL",
    "TWINKLE_PERSISTENCE_KEY_PREFIX",
)

# actor_name is a per-process cache key and is excluded from worker policy.
SERVER_STATE_ENV_KEYS: tuple[str, ...] = (
    "TWINKLE_SERVER_STATE_EXPIRATION_TIMEOUT",
    "TWINKLE_SERVER_STATE_CLEANUP_INTERVAL",
    "TWINKLE_SERVER_STATE_PER_TOKEN_MODEL_LIMIT",
    "TWINKLE_SERVER_STATE_METRICS_UPDATE_INTERVAL",
)

EXPLICIT_PROCESS_ENV_KEYS = ("PYTHONPATH", "TWINKLE_DEFAULT_SAVE_DIR", "TMPDIR", "OMP_NUM_THREADS")


def build_shared_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in SHARED_WORKER_ENV_KEYS if key in os.environ}


def build_persistence_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in PERSISTENCE_ENV_KEYS if key in os.environ}


def build_server_state_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in SERVER_STATE_ENV_KEYS if key in os.environ}


def build_propagated_env_vars() -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(build_shared_env_vars())
    merged.update(build_persistence_env_vars())
    merged.update(build_server_state_env_vars())
    return merged


def build_worker_env_vars() -> dict[str, str]:
    """Policy plus the explicit process settings shared by launcher and resource owners."""
    env = build_propagated_env_vars()
    for key in EXPLICIT_PROCESS_ENV_KEYS:
        if key in os.environ:
            env.setdefault(key, os.environ[key])
    return env
