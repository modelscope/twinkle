# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared instance startup, allocation, shutdown and monitoring budgets."""
import os
from pydantic import BaseModel, ConfigDict, Field


class ResourceOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    allocation_timeout_s: float = Field(default=3, gt=0, allow_inf_nan=False)
    startup_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)
    shutdown_timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    sweep_interval_s: float = Field(default=5, gt=0, allow_inf_nan=False)


SHARED_WORKER_ENV_KEYS: tuple[str, ...] = (
    'TWINKLE_STATE_RAY_NAMESPACE',
    'TWINKLE_TELEMETRY_ENABLED',
    'TWINKLE_TELEMETRY_DEBUG',
    'TWINKLE_TELEMETRY_SERVICE',
    'TWINKLE_TELEMETRY_ENDPOINT',
    'TWINKLE_TELEMETRY_INTERVAL',
    'TWINKLE_MODEL_ID_ALIASES',
)


def build_shared_env_vars() -> dict[str, str]:
    return {key: os.environ[key] for key in SHARED_WORKER_ENV_KEYS if key in os.environ}


def build_persistence_env_vars() -> dict[str, str]:
    from twinkle.server.config.persistence import PERSISTENCE_ENV_KEYS
    return {key: os.environ[key] for key in PERSISTENCE_ENV_KEYS if key in os.environ}


def build_server_state_env_vars() -> dict[str, str]:
    from twinkle.server.config.application_spec import SERVER_STATE_ENV_KEYS
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
    for key in ('PYTHONPATH', 'TWINKLE_DEFAULT_SAVE_DIR', 'TMPDIR', 'OMP_NUM_THREADS'):
        if key in os.environ:
            env.setdefault(key, os.environ[key])
    return env
