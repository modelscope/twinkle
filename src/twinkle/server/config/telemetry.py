# Copyright (c) ModelScope Contributors. All rights reserved.
"""Telemetry pipeline configuration model."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TelemetryConfig(BaseModel):
    """Configuration for the OpenTelemetry pipeline."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    service_name: str = 'twinkle-server'
    otlp_endpoint: str = 'http://localhost:4317'
    debug: bool = False  # True: Console Exporter; False: OTLP Exporter
    export_interval_ms: int = 30000
    resource_attributes: dict = Field(default_factory=dict)
