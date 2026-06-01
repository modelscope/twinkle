# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-deployment ``ApplicationSpec`` and typed argument schemas (R6, R3).

Each deployment kind (``server | model | sampler | processor``) carries its
own ``args`` block with strict field validation. ``ApplicationSpec`` holds
the routing metadata plus the deployment kind and validates ``args`` against
the matching ``*Args`` schema in a model validator.

The schemas use ``extra='forbid'`` so unknown args (typos, copy-paste from
other deployments) fail at load time instead of being silently dropped.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from twinkle.server.utils.task_queue.config import TaskQueueConfig


# ---------- shared helpers ------------------------------------------------- #


class _ArgsBase(BaseModel):
    """Base class for every per-deployment args schema (extra='forbid')."""

    model_config = ConfigDict(extra='forbid')


class HttpOptions(BaseModel):
    """HTTP listener settings (host/port).

    Re-exported from ``server_config`` for convenience and so that
    ``ServerArgs`` can carry it without importing the aggregate root.
    """

    model_config = ConfigDict(extra='forbid')

    host: str = 'localhost'
    port: int = 8000


# ---------- per-deployment args schemas (R3.x) ----------------------------- #


_BACKEND_VALUES: tuple[str, ...] = ('mock', 'transformers', 'megatron')
_SAMPLER_TYPE_VALUES: tuple[str, ...] = ('mock', 'vllm', 'torch')


class ModelArgs(_ArgsBase):
    """Args for the ``model`` deployment.

    The ``backend`` field selects the model implementation and replaces the
    legacy ``use_megatron: bool`` flag. Phase 0c introduces this field; the
    actual dispatch on its value is wired up in Phase 1 (R3.1-3.3, R3.9).
    """

    model_id: str
    nproc_per_node: int = 1
    device_group: dict[str, Any]
    device_mesh: dict[str, Any]
    backend: Literal['mock', 'transformers', 'megatron']
    adapter_config: dict[str, Any] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    max_loras: int = 5
    max_length: int | None = None


class SamplerArgs(_ArgsBase):
    """Args for the ``sampler`` deployment.

    ``sampler_type`` selects the sampler implementation (R3.4-3.6, R3.10).
    """

    model_id: str
    nproc_per_node: int = 1
    device_group: dict[str, Any]
    device_mesh: dict[str, Any]
    sampler_type: Literal['mock', 'vllm', 'torch']
    engine_args: dict[str, Any] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)


class ServerArgs(_ArgsBase):
    """Args for the gateway ``server`` deployment."""

    server_config: dict[str, Any] | None = None
    supported_models: list[Any] | None = None
    http_options: HttpOptions | None = None
    route_prefix: str | None = None


class ProcessorArgs(_ArgsBase):
    """Args for the ``processor`` deployment."""

    ncpu_proc_per_node: int | None = None
    device_group: dict[str, Any] | None = None
    device_mesh: dict[str, Any] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)


_ARGS_SCHEMA: dict[str, type[_ArgsBase]] = {
    'server': ServerArgs,
    'model': ModelArgs,
    'sampler': SamplerArgs,
    'processor': ProcessorArgs,
}


# ---------- ApplicationSpec ------------------------------------------------ #


class ApplicationSpec(BaseModel):
    """One application entry under ``ServerConfig.applications``.

    The ``args`` block is validated against the schema selected by
    ``import_path``: ``server`` → ``ServerArgs``, ``model`` → ``ModelArgs``,
    etc. Unknown keys at this level (or inside ``args``) are rejected.
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    route_prefix: str = '/'
    import_path: Literal['server', 'model', 'sampler', 'processor']
    args: ServerArgs | ModelArgs | SamplerArgs | ProcessorArgs = Field(
        default_factory=lambda: ServerArgs(),
    )
    deployments: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _coerce_args_to_schema(cls, data: Any) -> Any:
        """Validate the raw ``args`` dict against the schema for ``import_path``.

        Pydantic's union resolution would also work here, but keying off
        ``import_path`` makes the failure messages point at the right schema
        and avoids ambiguity when two schemas share field names.
        """
        if not isinstance(data, dict):
            return data
        import_path = data.get('import_path')
        args = data.get('args')
        if import_path in _ARGS_SCHEMA and isinstance(args, dict):
            schema = _ARGS_SCHEMA[import_path]
            data = {**data, 'args': schema.model_validate(args)}
        return data
