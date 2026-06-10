# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-deployment ``ApplicationSpec`` and typed argument schemas.

Each deployment kind (``server | model | sampler | processor``) carries its
own ``args`` block with strict field validation. ``ApplicationSpec`` holds
the routing metadata plus the deployment kind and validates ``args`` against
the matching ``*Args`` schema in a model validator.

The schemas use ``extra='forbid'`` so unknown args (typos, copy-paste from
other deployments) fail at load time instead of being silently dropped.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Literal

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


# ---------- per-deployment args schemas ------------------------------------ #


class ModelArgs(_ArgsBase):
    """Args for the ``model`` deployment.

    The ``backend`` field selects the model implementation (transformers /
    megatron / mock) — see ``model/app.py`` for the dispatch site that picks
    the runtime class based on this value.
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

    ``sampler_type`` selects the sampler implementation.
    """

    model_id: str
    nproc_per_node: int = 1
    device_group: dict[str, Any]
    device_mesh: dict[str, Any]
    sampler_type: Literal['mock', 'vllm', 'torch']
    engine_args: dict[str, Any] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)


class ServerStateArgs(_ArgsBase):
    """Typed args for the gateway's ``ServerState`` configuration.

    Replaces the former untyped ``server_config: dict[str, Any] | None`` so a
    misspelled key fails validation (``extra='forbid'``) instead of being
    silently forwarded. Named ``ServerStateArgs`` to avoid colliding with the
    ``ServerConfig`` aggregate root. All fields are optional (default ``None``)
    so unset values fall back to ``ServerState``'s own defaults; the
    operator-facing YAML key stays ``server_config``.
    """

    expiration_timeout: float | None = None
    cleanup_interval: float | None = None
    per_token_model_limit: int | None = None
    metrics_update_interval: float | None = None
    actor_name: str | None = None


class ServerArgs(_ArgsBase):
    """Args for the gateway ``server`` deployment."""

    server_config: ServerStateArgs = Field(default_factory=ServerStateArgs)
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
    etc. Unknown keys at this level (or inside ``args``) are rejected. A
    missing ``args`` block is treated as ``{}`` and validated against the
    matching schema, so any required field (e.g. ``backend`` on a model
    deployment) raises with the offending field path instead of silently
    falling back to a different schema's default.
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    route_prefix: str = '/'
    import_path: Literal['server', 'model', 'sampler', 'processor']
    # ``args`` is always populated by the ``mode='before'`` validator below
    # (which validates the raw block against the schema selected by
    # ``import_path`` and defaults a missing block to ``{}``), so the field is
    # required here — the validator runs first and fills it.
    args: ServerArgs | ModelArgs | SamplerArgs | ProcessorArgs
    deployments: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _coerce_args_to_schema(cls, data: Any) -> Any:
        """Validate the raw ``args`` block against the schema for ``import_path``.

        Keying off ``import_path`` makes the failure messages point at the
        right schema and avoids the ambiguity Pydantic's structural Union
        resolution would introduce when two schemas share field names.
        """
        if not isinstance(data, dict):
            return data
        import_path = data.get('import_path')
        if import_path not in _ARGS_SCHEMA:
            # Let Pydantic's Literal validator handle bad import_path values.
            return data
        schema = _ARGS_SCHEMA[import_path]
        raw_args = data.get('args')
        if isinstance(raw_args, schema):
            return data
        if raw_args is None:
            raw_args = {}
        # ``schema.model_validate`` rejects a non-dict itself with a clean
        # error, so no separate non-dict guard is needed here.
        return {**data, 'args': schema.model_validate(raw_args)}
