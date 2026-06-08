# Copyright (c) ModelScope Contributors. All rights reserved.
"""Server configuration package — aggregate root and per-deployment specs."""

from .application_spec import ApplicationSpec, HttpOptions, ModelArgs, ProcessorArgs, SamplerArgs, ServerArgs
from .persistence import PersistenceConfig
from .server_config import ServerConfig
from .telemetry import TelemetryConfig

__all__ = [
    'ApplicationSpec',
    'HttpOptions',
    'ModelArgs',
    'PersistenceConfig',
    'ProcessorArgs',
    'SamplerArgs',
    'ServerArgs',
    'ServerConfig',
    'TelemetryConfig',
]
