# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training YAML arguments and the controller's validated runtime configuration."""
from __future__ import annotations

import math
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, ValidationInfo, field_validator, model_validator
from typing import TYPE_CHECKING, Any, Literal

from twinkle.hub.model_alias import extract_model_alias
from twinkle.server.exceptions import RequestRejectedError
from .resources import ResourceOptions
from .task_queue import TaskQueueConfig

if TYPE_CHECKING:
    from .application_spec import ApplicationSpec


class SpectralHybridArgs(BaseModel):
    """Strict server-owned Spectral Hybrid allocation and optimizer defaults."""

    model_config = ConfigDict(extra='forbid')

    allocation_path: str = Field(min_length=1)
    default_lr_lora: float = Field(default=2.5e-5, gt=0)
    default_lr_fft: float = Field(default=1.0e-6, gt=0)

    @field_validator('default_lr_lora', 'default_lr_fft')
    @classmethod
    def _finite_learning_rate(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError('learning rate must be finite')
        return value


class AdapterConfig(BaseModel):
    """Session liveness and absolute lifetime of a training handle, in seconds."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    adapter_timeout: float | None = Field(default=1800, gt=0, allow_inf_nan=False)
    adapter_max_lifetime: float | None = Field(default=36000, gt=0, allow_inf_nan=False)


class TrainingArguments(BaseModel):
    """Training options shared by model YAML and resolved instance configuration."""

    model_config = ConfigDict(extra='forbid')

    min_replicas: int = Field(default=0, ge=0, strict=True)
    max_replicas: int | None = Field(default=None, gt=0, strict=True)
    idle_timeout_s: float = Field(default=1800, gt=0, allow_inf_nan=False)
    adapter_config: AdapterConfig = Field(default_factory=AdapterConfig)
    strategy: Literal['accelerate', 'native_fsdp'] | None = None
    queue_config: TaskQueueConfig = Field(default_factory=TaskQueueConfig)
    max_length: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode='after')
    def _validate_replicas(self):
        if self.max_replicas is not None and self.min_replicas > self.max_replicas:
            raise ValueError('min_replicas must not exceed max_replicas')
        return self


class LoraArguments(BaseModel):
    """LoRA parameters shared by YAML and LoRA instance configuration."""

    max_loras: int = Field(default=5, gt=0, strict=True)
    max_r: int = Field(default=32, gt=0, strict=True)
    target_modules: str | list[str] = 'all-linear'
    hybrid: SpectralHybridArgs | None = None


class ModelArgs(TrainingArguments, LoraArguments):
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
    env_vars: dict[str, str] = Field(default_factory=dict)
    train_mode: Literal['lora', 'hybrid', 'full'] = 'lora'
    data_plane_url: str | None = None

    @model_validator(mode='after')
    def _validate_spectral_backend(self):
        if self.strategy is not None and self.backend != 'transformers':
            raise ValueError('strategy is only supported by the transformers backend')
        if self.hybrid is not None:
            if self.backend != 'transformers':
                raise ValueError('hybrid is only supported by the transformers backend')
            if self.train_mode == 'full':
                raise ValueError('hybrid is not supported with train_mode="full"')
        return self


class TrainingMesh(BaseModel):
    """Sizes accepted by DeviceMesh.from_sizes, including overlapping EP axes."""

    model_config = ConfigDict(extra='forbid', frozen=True)

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


class TrainingModeConfig(TrainingArguments):
    model_config = ConfigDict(extra='forbid', frozen=True)

    num_devices: int = Field(gt=0, strict=True)
    device_mesh: TrainingMesh

    @model_validator(mode='after')
    def _validate_resources(self):
        if self.device_mesh.world_size() != self.num_devices:
            raise ValueError('device_mesh world_size must equal num_devices')
        return self


class LoraTrainingConfig(TrainingModeConfig, LoraArguments):
    pass


class TrainingModelConfig(BaseModel):
    """One application exposes one training mode and one instance specification."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    backend: Literal['mock', 'transformers', 'megatron']
    device_type: Literal['cuda', 'npu', 'mps', 'cpu']
    mode: Literal['lora', 'full']
    config: SerializeAsAny[TrainingModeConfig]
    route_prefix: str | None = None
    data_plane_url: str | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    deploy_options: dict = Field(default_factory=dict)

    @field_validator('config', mode='before')
    @classmethod
    def _instance_config(cls, value, info: ValidationInfo):
        config_type = LoraTrainingConfig if info.data.get('mode') == 'lora' else TrainingModeConfig
        data = value.model_dump() if isinstance(value, TrainingModeConfig) else value
        return config_type.model_validate(data)

    @model_validator(mode='after')
    def _validate_backend(self):
        if (self.name.startswith('/') or any(part in ('', '.', '..') for part in self.name.split('/'))
                or any(char.isspace() or char in '?#%\\' for char in self.name)):
            raise ValueError('name must be a nonempty relative model path without dot segments')
        if self.device_type == 'cpu' and self.backend != 'mock':
            raise ValueError('cpu is supported only by the mock training backend')
        if self.backend == 'megatron' and self.config.num_devices < 2:
            raise ValueError('megatron training instances require at least 2 devices')
        if self.config.device_mesh.device_type not in (None, self.device_type):
            raise ValueError('device_mesh.device_type must match device_type')
        if self.config.strategy is not None and self.backend != 'transformers':
            raise ValueError('strategy requires the transformers backend')
        if isinstance(self.config, LoraTrainingConfig) and self.config.hybrid and self.backend != 'transformers':
            raise ValueError('hybrid requires the transformers backend')
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.name, self.mode

    @property
    def capacity(self) -> int:
        return self.config.max_loras if isinstance(self.config, LoraTrainingConfig) else 1

    @classmethod
    def from_application(cls, app: ApplicationSpec) -> TrainingModelConfig:
        args = app.args
        if not isinstance(args, ModelArgs):
            raise TypeError('Training configuration requires model application arguments')
        if len(app.deployments) > 1:
            raise ValueError(f'{app.name}: a model application must have one deployment configuration')
        deployment = app.deployments[0] if app.deployments else {}
        device_type = str(
            args.device_group.get('device_type',
                                  args.device_mesh.get('device_type',
                                                       'cpu' if args.backend == 'mock' else 'cuda'))).lower()
        mode = 'full' if args.train_mode == 'full' else 'lora'
        fields = set(TrainingArguments.model_fields)
        if mode == 'lora':
            fields.update(LoraArguments.model_fields)
        config = {key: getattr(args, key) for key in fields}
        config.update(
            num_devices=args.device_group.get('ranks', args.nproc_per_node),
            device_mesh={
                **args.device_mesh, 'device_type': device_type
            })
        env_vars = {
            **(deployment.get('ray_actor_options') or {}).get('runtime_env', {}).get('env_vars', {}),
            **args.env_vars,
        }
        return cls(
            name=extract_model_alias(app.route_prefix) or app.name,
            model_id=args.model_id,
            backend=args.backend,
            device_type=device_type,
            mode=mode,
            config=config,
            route_prefix=app.route_prefix.rstrip('/') or '/',
            data_plane_url=args.data_plane_url,
            env_vars=env_vars,
            deploy_options={
                key: value
                for key, value in deployment.items() if key != 'name'
            })


class TrainingConfig(ResourceOptions):
    """Resolved catalog; public model entries stay in applications."""

    models: tuple[TrainingModelConfig, ...] = ()

    @model_validator(mode='after')
    def _unique_names(self):
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError('training model names must be unique')
        return self

    def get(self, name: str, mode: Literal['lora', 'full']) -> TrainingModelConfig:
        """Resolve a declared specification without allocating resources."""
        model = next((model for model in self.models if model.name == name), None)
        if model is None:
            raise RequestRejectedError(f'Unknown training model: {name}')
        if model.mode != mode:
            raise RequestRejectedError(f'Training model {name} does not provide {mode} mode')
        return model
