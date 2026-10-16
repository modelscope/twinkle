# Copyright (c) ModelScope Contributors. All rights reserved.
"""Administrator-owned training catalog. Parsing never allocates devices."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from twinkle.server.task_queue.config import TaskQueueConfig
from .hybrid import SpectralHybridArgs


class _TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class TrainingMesh(_TrainingConfig):
    """Sizes accepted by DeviceMesh.from_sizes, including overlapping EP axes."""

    device_type: str | None = None
    dp_size: int = Field(default=1, gt=0, strict=True)
    fsdp_size: int | None = Field(default=None, gt=0, strict=True)
    tp_size: int | None = Field(default=None, gt=0, strict=True)
    pp_size: int | None = Field(default=None, gt=0, strict=True)
    cp_size: int | None = Field(default=None, gt=0, strict=True)
    ep_size: int | None = Field(default=None, gt=0, strict=True)
    etp_size: int = Field(default=1, gt=0, strict=True)
    ep_fsdp_size: int | None = Field(default=None, gt=0, strict=True)
    vpp_size: int | None = Field(default=None, gt=0, strict=True)
    ulysses_size: int | None = Field(default=None, gt=0, strict=True)
    sequence_parallel: bool = False

    def world_size(self) -> int:
        from twinkle.utils.device_mesh import DeviceMesh
        return DeviceMesh.from_sizes(**self.model_dump(exclude_none=True)).world_size


class TrainingModeConfig(_TrainingConfig):
    num_devices: int = Field(gt=0, strict=True)
    device_mesh: TrainingMesh
    min_replicas: int = Field(default=0, ge=0, strict=True)
    max_replicas: int | None = Field(default=None, gt=0, strict=True)
    idle_timeout_s: float | None = Field(default=1800, gt=0, allow_inf_nan=False)
    strategy: Literal['accelerate', 'native_fsdp'] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    max_length: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode='after')
    def _validate_resources(self):
        if self.max_replicas is not None and self.min_replicas > self.max_replicas:
            raise ValueError('min_replicas must not exceed max_replicas')
        if self.device_mesh.world_size() != self.num_devices:
            raise ValueError('device_mesh world_size must equal num_devices')
        return self


class LoraTrainingConfig(TrainingModeConfig):
    max_loras: int = Field(default=5, gt=0, strict=True)
    max_r: int = Field(default=32, gt=0, strict=True)
    target_modules: str | list[str] = 'all-linear'
    hybrid: SpectralHybridArgs | None = None


class TrainingModes(_TrainingConfig):
    lora: LoraTrainingConfig | None = None
    full: TrainingModeConfig | None = None

    @model_validator(mode='after')
    def _not_empty(self):
        if self.lora is None and self.full is None:
            raise ValueError('at least one training mode is required')
        return self


class LegacyExpiration(_TrainingConfig):
    """Internal compatibility policy derived from existing adapter_config fields."""
    session_timeout_s: float | None = Field(default=1800, gt=0, allow_inf_nan=False)
    max_lifetime_s: float | None = Field(default=36000, gt=0, allow_inf_nan=False)


class TrainingModelConfig(_TrainingConfig):
    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    backend: Literal['mock', 'transformers', 'megatron']
    device_type: Literal['cuda', 'npu', 'mps', 'cpu']
    modes: TrainingModes
    legacy_expiration: LegacyExpiration | None = None
    route_prefix: str | None = None
    data_plane_url: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _validate_backend(self):
        if (self.name.startswith('/') or any(part in ('', '.', '..') for part in self.name.split('/'))
                or any(char.isspace() or char in '?#%\\' for char in self.name)):
            raise ValueError('name must be a nonempty relative model path without dot segments')
        if self.device_type == 'cpu' and self.backend != 'mock':
            raise ValueError('cpu is supported only by the mock training backend')
        for mode in (self.modes.lora, self.modes.full):
            if mode is None:
                continue
            if mode.device_mesh.device_type not in (None, self.device_type):
                raise ValueError('device_mesh.device_type must match device_type')
            if mode.strategy is not None and self.backend != 'transformers':
                raise ValueError('strategy requires the transformers backend')
        if self.modes.lora and self.modes.lora.hybrid and self.backend != 'transformers':
            raise ValueError('hybrid requires the transformers backend')
        return self


class TrainingOptions(_TrainingConfig):
    allocation_timeout_s: float = Field(default=3, gt=0, allow_inf_nan=False)
    startup_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)
    shutdown_timeout_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    sweep_interval_s: float = Field(default=5, gt=0, allow_inf_nan=False)


class TrainingConfig(TrainingOptions):
    """Internal normalized catalog; public model entries stay in applications."""

    models: tuple[TrainingModelConfig, ...] = ()

    @model_validator(mode='after')
    def _unique_names(self):
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError('training model names must be unique')
        return self
