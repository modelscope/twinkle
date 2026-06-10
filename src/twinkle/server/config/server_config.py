# Copyright (c) ModelScope Contributors. All rights reserved.
"""Aggregate-root server configuration.

``ServerConfig`` is the single Pydantic model that nests every configuration
subsystem the launcher consumes (telemetry, persistence, task-queue, and the
list of ``ApplicationSpec``). Loading a YAML file and validating it goes
through one entry point — ``ServerConfig.from_yaml(path)`` — so the launcher
no longer reaches into a raw dict.

Top-level fields use their current names with no aliases for legacy names
(``telemetry_config``, ``persistence_config``); the model is configured with
``extra='forbid'`` so a YAML that uses a legacy field is rejected with the
offending name pointed at.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any

from twinkle.server.exceptions import ConfigParseError
from twinkle.server.utils.task_queue.config import TaskQueueConfig
from .application_spec import ApplicationSpec, HttpOptions
from .persistence import PersistenceConfig
from .telemetry import TelemetryConfig


class ServerConfig(BaseModel):
    """Top-level server configuration aggregate root."""

    model_config = ConfigDict(extra='forbid')

    ray_namespace: str | None = None
    proxy_location: str | None = None
    http_options: HttpOptions = Field(default_factory=HttpOptions)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    task_queue: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    applications: list[ApplicationSpec] = Field(default_factory=list)

    # ---- loading ---------------------------------------------------------- #

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServerConfig:
        """Load and validate a YAML file into a ``ServerConfig``.

        Raises:
            FileNotFoundError: ``path`` does not exist or cannot be read.
            ConfigParseError: ``path`` exists but is not well-formed YAML.
            pydantic.ValidationError: a field or cross-field constraint
                fails — the error names every offending field.
        """
        from omegaconf import OmegaConf

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Config file not found: {p}')
        try:
            raw = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
        except Exception as e:  # malformed YAML / OmegaConf parse failure
            raise ConfigParseError(f'Malformed YAML in {p}: {e}') from e
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigParseError(f'Top-level YAML in {p} must be a mapping, got {type(raw).__name__}', )
        return cls.model_validate(raw)

    # ---- cross-field validation ------------------------------------------ #

    @model_validator(mode='after')
    def _validate_cross_field(self) -> ServerConfig:
        if self.persistence.mode == 'redis' and not self.persistence.redis_url:
            raise ValueError("persistence.redis_url is required when persistence.mode == 'redis'", )
        if self.persistence.mode == 'file' and not self.persistence.file_path:
            raise ValueError("persistence.file_path is required when persistence.mode == 'file'", )
        return self

    # ---- round-trip / serialization -------------------------------------- #

    def to_yaml_dict(self) -> dict[str, Any]:
        """Return a JSON-mode dict suitable for ``yaml.safe_dump`` / round-trip."""
        return self.model_dump(mode='json')
