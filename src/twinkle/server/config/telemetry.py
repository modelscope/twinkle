# Copyright (c) ModelScope Contributors. All rights reserved.
"""Telemetry pipeline configuration model."""
from __future__ import annotations
import os

import json
from pydantic import BaseModel, ConfigDict, Field

_WORKER_ENV = {
    'enabled': 'TWINKLE_TELEMETRY_ENABLED',
    'debug': 'TWINKLE_TELEMETRY_DEBUG',
    'service_name': 'TWINKLE_TELEMETRY_SERVICE',
    'otlp_endpoint': 'TWINKLE_TELEMETRY_ENDPOINT',
    'export_interval_ms': 'TWINKLE_TELEMETRY_INTERVAL',
    'resource_attributes': 'TWINKLE_TELEMETRY_RESOURCE_ATTRIBUTES',
}
TELEMETRY_WORKER_ENV_KEYS = tuple(_WORKER_ENV.values())


class TelemetryConfig(BaseModel):
    """Configuration for the OpenTelemetry pipeline."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    service_name: str = 'twinkle-server'
    otlp_endpoint: str = 'http://localhost:4317'
    debug: bool = False  # True: Console Exporter; False: OTLP Exporter
    export_interval_ms: int = Field(default=30000, gt=0)
    resource_attributes: dict = Field(default_factory=dict)

    def to_env_vars(self) -> dict[str, str]:
        """Pass the resolved configuration through Ray's string-valued environment."""
        values = self.model_dump()
        values.update(enabled=int(self.enabled), debug=int(self.debug),
                      resource_attributes=json.dumps(self.resource_attributes))
        return {key: str(values[field]) for field, key in _WORKER_ENV.items()}

    @classmethod
    def from_env(cls) -> TelemetryConfig:
        if os.environ.get(_WORKER_ENV['enabled']) != '1':
            return cls()
        values = {field: os.environ[key] for field, key in _WORKER_ENV.items() if key in os.environ}
        if 'debug' in values:
            values['debug'] = values['debug'] == '1'
        if 'resource_attributes' in values:
            values['resource_attributes'] = json.loads(values['resource_attributes'])
        return cls.model_validate(values)
