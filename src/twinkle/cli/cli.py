# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, Union


# ────────────────────────────────────────────────────────────────────────────────
# Arg group dataclasses
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelArgs:
    model_id: Optional[str] = field(default=None, metadata={'primary': True})
    model_cls: Optional[str] = None
    tokenizer_id: Optional[str] = None
    mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16'
    strategy: Literal['accelerate', 'native_fsdp'] = field(
        default='accelerate', metadata={'aliases': ('use_megatron',)})
    memory_efficient_init: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True
    ddp_config: Optional[Dict[str, Any]] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    grad_scaler_config: Optional[Dict[str, Any]] = None


@dataclass
class LoraArgs:
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    adapter_name: str = 'default'


@dataclass
class DatasetArgs:
    dataset_id: str = ''
    subset_name: str = 'default'
    split: str = 'train'
    streaming: bool = False
    num_proc: Optional[int] = None
    data_slice: Optional[str] = None
    revision: Optional[str] = None


@dataclass
class TemplateArgs:
    template_cls: Optional[str] = None
    model_id: Optional[str] = None
    max_length: int = 8192
    truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise'
    use_chat_template: bool = True
    enable_thinking: bool = True
    default_system: Optional[str] = None


@dataclass
class TrainingArgs:
    max_steps: int = 200
    num_train_epochs: Optional[int] = None
    batch_size: int = 8
    mini_batch_size: Optional[int] = None
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    output_dir: str = './output'
    save_steps: int = 50
    save_total_limit: Optional[int] = None
    log_interval: int = 10
    eval_interval: Optional[int] = None
    eval_samples: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    ignore_data_skip: bool = False
    seed: int = field(default=42, metadata={'primary': True})
    full_determinism: bool = False
    padding_free: bool = False


@dataclass
class OptimizerArgs:
    optimizer_cls: str = 'AdamW'
    learning_rate: float = field(default=1e-5, metadata={'aliases': ('lr',)})
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class SchedulerArgs:
    scheduler_cls: str = 'CosineAnnealingLR'
    num_warmup_steps: int = 0
    num_training_steps: Optional[int] = None
    t_max: Optional[int] = None
    eta_min: float = 0.0
    lr_decay_steps: Optional[int] = None
    max_lr: Optional[float] = None


@dataclass
class LossArgs:
    loss_cls: str = 'GRPOLoss'
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    beta: float = 0.0
    entropy_coef: float = 0.0
    ignore_index: int = -100


@dataclass
class SamplerArgs:
    sampler_type: str = 'vLLMSampler'
    gpu_memory_utilization: float = 0.8
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    enable_lora: bool = False
    max_lora_rank: int = 32
    enforce_eager: bool = False


@dataclass
class SamplingArgs:
    max_tokens: Optional[int] = field(default=None, metadata={'aliases': ('max_new_tokens',)})
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_samples: int = 1
    logprobs: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[str] = None


@dataclass
class InfraArgs:
    mode: Literal['local', 'ray'] = 'local'
    nproc_per_node: int = field(default=8, metadata={'aliases': ('num_gpus',)})
    ncpu_proc_per_node: int = 8
    model_gpus: Optional[int] = None
    sampler_gpus: Optional[int] = None
    dp_size: Optional[int] = None
    fsdp_size: Optional[int] = None
    tp_size: Optional[int] = None
    cp_size: Optional[int] = None
    ep_size: Optional[int] = None
    ulysses_size: Optional[int] = None
    lazy_collect: bool = True


@dataclass
class ServerArgs:
    config: Optional[str] = None
    ray_namespace: str = 'twinkle_cluster'
    host: str = '0.0.0.0'
    port: int = 8000
    log_level: str = 'INFO'


@dataclass
class RLArgs:
    num_generations: int = 8
    advantage_type: str = 'GRPOAdvantage'
    advantage_scale: Literal['group', 'batch', 'none'] = 'group'
    reward_fns: Optional[List[str]] = None


@dataclass
class CheckpointArgs:
    save_optimizer: bool = True
    merge_and_sync: bool = True
    platform: str = 'GPU'


# ────────────────────────────────────────────────────────────────────────────────
# ConfigSource hierarchy
# ────────────────────────────────────────────────────────────────────────────────


class ConfigSource(ABC):
    """Base class for all configuration sources."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Return raw key-value pairs from this source."""
        ...


class DotEnvSource(ConfigSource):

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._path = path

    def load(self) -> Dict[str, str]:
        path = self._resolve_path()
        if path is None:
            return {}
        result: Dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def _resolve_path(self) -> Optional[Path]:
        if self._path is not None:
            p = Path(self._path)
            return p if p.is_file() else None
        for name in ('.env', '.env.local'):
            p = Path.cwd() / name
            if p.is_file():
                return p
        return None


class EnvVarSource(ConfigSource):
    """Reads os.environ; recognizes TWINKLE_ prefix and any key known to the registry."""

    def __init__(self, registry: 'ConfigRegistry'):
        self._registry = registry

    def load(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith('TWINKLE_'):
                result[key[8:]] = value
            elif self._registry.resolve(key) is not None:
                result[key] = value
        return result


class YamlSource(ConfigSource):

    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)

    def load(self) -> Dict[str, Any]:
        from omegaconf import OmegaConf
        if not self._path.is_file():
            raise FileNotFoundError(f'Config file not found: {self._path}')
        cfg = OmegaConf.load(self._path)
        return OmegaConf.to_container(cfg, resolve=True)


class CLISource(ConfigSource):

    def __init__(self, argv: Optional[List[str]] = None):
        self._argv = argv if argv is not None else sys.argv[1:]

    def load(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        i = 0
        argv = self._argv
        while i < len(argv):
            token = argv[i]
            if not token.startswith('--'):
                i += 1
                continue
            token = token[2:]
            if token.startswith('no_') or token.startswith('no-'):
                result[token[3:]] = False
                i += 1
                continue
            if '=' in token:
                key, _, value = token.partition('=')
                result[key] = value
                i += 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith('--'):
                result[token] = argv[i + 1]
                i += 2
            else:
                result[token] = True
                i += 1
        return result


# ────────────────────────────────────────────────────────────────────────────────
# ConfigRegistry: maps normalized keys to (group_name, field_name)
# ────────────────────────────────────────────────────────────────────────────────


class ConfigRegistry:
    """Introspects Args dataclass groups to build a case-insensitive key→field map."""

    # Same field name in 2+ groups — the winning group must declare metadata={'primary': True}

    def __init__(self, groups: Dict[str, Any]):
        self._field_map: Dict[str, Tuple[str, str]] = {}
        self._alias_map: Dict[str, str] = {}
        self._groups = groups
        self._build(groups)

    def _build(self, groups: Dict[str, Any]) -> None:
        owners: Dict[str, List[Tuple[str, bool]]] = {}
        for group_name, group_obj in groups.items():
            for f in fields(group_obj):
                is_primary = f.metadata.get('primary', False)
                owners.setdefault(f.name.lower(), []).append((group_name, is_primary))
                for alias in f.metadata.get('aliases', ()):  # field-local aliases
                    self._alias_map[alias.lower()] = f.name.lower()
        for key, owner_list in owners.items():
            if len(owner_list) == 1:
                self._field_map[key] = (owner_list[0][0], key)
                continue
            primaries = [g for g, p in owner_list if p]
            if len(primaries) != 1:
                all_groups = [g for g, _ in owner_list]
                raise ValueError(
                    f'Field {key!r} exists in groups {all_groups}; '
                    f"exactly one must declare metadata={{'primary': True}}, found {len(primaries)}")
            self._field_map[key] = (primaries[0], key)

    def resolve(self, key: str) -> Optional[Tuple[str, str]]:
        normalized = key.lower().replace('-', '_')
        canonical = self._alias_map.get(normalized, normalized)
        if canonical in self._field_map:
            return self._field_map[canonical]
        # prefix-based fallback: model_xxx → group=model, field=xxx
        for group_name in self._groups:
            prefix = group_name + '_'
            if canonical.startswith(prefix):
                stripped = canonical[len(prefix):]
                if stripped and (group_name, stripped) in (
                    (g, f.name) for g, obj in self._groups.items() for f in fields(obj)
                ):
                    return (group_name, stripped)
        return None

    def all_keys(self) -> Iterator[str]:
        return iter(self._field_map)


# ────────────────────────────────────────────────────────────────────────────────
# Args: unified container
# ────────────────────────────────────────────────────────────────────────────────


@dataclass
class Args:
    """Unified argument container. Access groups directly or via get_*_args() dicts."""

    model: ModelArgs = field(default_factory=ModelArgs)
    lora: LoraArgs = field(default_factory=LoraArgs)
    dataset: DatasetArgs = field(default_factory=DatasetArgs)
    template: TemplateArgs = field(default_factory=TemplateArgs)
    training: TrainingArgs = field(default_factory=TrainingArgs)
    optimizer: OptimizerArgs = field(default_factory=OptimizerArgs)
    scheduler: SchedulerArgs = field(default_factory=SchedulerArgs)
    loss: LossArgs = field(default_factory=LossArgs)
    sampler: SamplerArgs = field(default_factory=SamplerArgs)
    sampling: SamplingArgs = field(default_factory=SamplingArgs)
    infra: InfraArgs = field(default_factory=InfraArgs)
    server: ServerArgs = field(default_factory=ServerArgs)
    rl: RLArgs = field(default_factory=RLArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_model_args(self) -> Dict[str, Any]:
        d = self._to_dict(self.model)
        if not d.get('model_id') and self.template.model_id:
            d['model_id'] = self.template.model_id
        return d

    def get_lora_args(self) -> Dict[str, Any]:
        return {
            'target_modules': self.lora.lora_target_modules or 'all-linear',
            'r': self.lora.lora_r,
            'lora_alpha': self.lora.lora_alpha,
            'lora_dropout': self.lora.lora_dropout,
        }

    def get_dataset_args(self) -> Dict[str, Any]:
        return self._to_dict(self.dataset)

    def get_template_args(self) -> Dict[str, Any]:
        d = self._to_dict(self.template)
        if not d.get('model_id') and self.model.model_id:
            d['model_id'] = self.model.model_id
        return d

    def get_training_args(self) -> Dict[str, Any]:
        return self._to_dict(self.training)

    def get_optimizer_args(self) -> Dict[str, Any]:
        d = self._to_dict(self.optimizer)
        d['lr'] = d.pop('learning_rate', 1e-5)
        return d

    def get_scheduler_args(self) -> Dict[str, Any]:
        return self._to_dict(self.scheduler)

    def get_loss_args(self) -> Dict[str, Any]:
        return self._to_dict(self.loss)

    def get_sampler_args(self) -> Dict[str, Any]:
        return self._to_dict(self.sampler)

    def get_sampling_args(self) -> Dict[str, Any]:
        return self._to_dict(self.sampling)

    def get_infra_args(self) -> Dict[str, Any]:
        return self._to_dict(self.infra)

    def get_server_args(self) -> Dict[str, Any]:
        return self._to_dict(self.server)

    def get_rl_args(self) -> Dict[str, Any]:
        return self._to_dict(self.rl)

    def get_checkpoint_args(self) -> Dict[str, Any]:
        return self._to_dict(self.checkpoint)

    def get(self, key: str, default: Any = None) -> Any:
        for f in fields(self):
            if f.name == 'extra':
                continue
            group = getattr(self, f.name)
            if hasattr(group, key):
                return getattr(group, key)
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        val = self.get(key, _SENTINEL)
        if val is _SENTINEL:
            raise KeyError(key)
        return val

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            if f.name == 'extra':
                continue
            result.update(self._to_dict(getattr(self, f.name)))
        result.update(self.extra)
        return result

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        return {f.name: getattr(obj, f.name) for f in fields(obj) if getattr(obj, f.name) is not None}


_SENTINEL = object()


# ────────────────────────────────────────────────────────────────────────────────
# ValueCaster: type coercion
# ────────────────────────────────────────────────────────────────────────────────


class ValueCaster:

    @staticmethod
    def auto_cast(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        low = value.lower()
        if low in ('true', 'yes', 'on'):
            return True
        if low in ('false', 'no', 'off'):
            return False
        if low in ('none', 'null', '~'):
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        if ',' in value:
            return [ValueCaster.auto_cast(v.strip()) for v in value.split(',')]
        return value

    @staticmethod
    def coerce_to_field(obj: Any, field_name: str, value: Any) -> Any:
        current = getattr(obj, field_name, None)
        if current is None or value is None:
            return value
        target_type = type(current)
        if target_type is bool:
            if isinstance(value, bool):
                return value
            return ValueCaster.auto_cast(str(value))
        if target_type is int and not isinstance(value, int):
            try:
                return int(float(value)) if isinstance(value, str) else int(value)
            except (ValueError, TypeError):
                return value
        if target_type is float and not isinstance(value, (int, float)):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if target_type is list and isinstance(value, str):
            return [v.strip() for v in value.split(',')]
        return value


# ────────────────────────────────────────────────────────────────────────────────
# ConfigResolver: merges sources
# ────────────────────────────────────────────────────────────────────────────────


class ConfigResolver:

    def __init__(self, args: Args):
        self._args = args
        self._groups = {
            f.name: getattr(args, f.name)
            for f in fields(args)
            if f.name != 'extra'
        }
        self._registry = ConfigRegistry(self._groups)

    @property
    def registry(self) -> 'ConfigRegistry':
        return self._registry

    def apply(self, source: Dict[str, Any], cast_strings: bool = False) -> None:
        flat = self._flatten(source)
        for raw_key, raw_value in flat.items():
            key = raw_key.lower().replace('-', '_')
            value = ValueCaster.auto_cast(raw_value) if cast_strings else raw_value
            # handle use_megatron alias
            if key == 'use_megatron':
                if ValueCaster.auto_cast(str(value)):
                    self._set('model', 'strategy', 'native_fsdp')
                continue
            resolved = self._registry.resolve(key)
            if resolved:
                group_name, field_name = resolved
                group = self._groups[group_name]
                coerced = ValueCaster.coerce_to_field(group, field_name, value)
                setattr(group, field_name, coerced)
            else:
                self._args.extra[key] = value

    def _set(self, group_name: str, field_name: str, value: Any) -> None:
        group = self._groups[group_name]
        setattr(group, field_name, value)

    def _flatten(self, d: Any, prefix: str = '') -> Dict[str, Any]:
        if not isinstance(d, dict):
            return {prefix: d} if prefix else {}
        result: Dict[str, Any] = {}
        for key, value in d.items():
            full_key = f'{prefix}_{key}' if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten(value, full_key))
            else:
                result[full_key] = value
        return result


# ────────────────────────────────────────────────────────────────────────────────
# CLI: top-level entry point
# ────────────────────────────────────────────────────────────────────────────────


class CLI:
    """Unified configuration parser.

    Resolution order (later wins):
        1. Dataclass defaults
        2. .env file
        3. Environment variables (TWINKLE_ prefix or bare)
        4. YAML config file (--config / explicit)
        5. CLI overrides (--key value)

    All keys are case-insensitive and dash/underscore equivalent:
        --model-id, MODEL_ID, TWINKLE_MODEL_ID, model_id: in .yaml all resolve the same.
    """

    @staticmethod
    def from_args(
        argv: Optional[List[str]] = None,
        env_file: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
    ) -> Args:
        args = Args()
        resolver = ConfigResolver(args)

        # 1. .env
        resolver.apply(DotEnvSource(env_file).load(), cast_strings=True)

        # 2. Environment variables
        resolver.apply(EnvVarSource(resolver.registry).load(), cast_strings=True)

        # 3. CLI (first pass to extract --config)
        cli_data = CLISource(argv).load()
        yaml_path = config_file or cli_data.pop('config', None)

        # 4. YAML
        if yaml_path:
            resolver.apply(YamlSource(yaml_path).load(), cast_strings=False)

        # 5. CLI overrides (highest priority, values are strings from argv)
        resolver.apply(cli_data, cast_strings=True)

        return args
