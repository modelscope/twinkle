# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared configuration helpers for synchronous and asynchronous RL runners."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from twinkle.data_format import SampleResponse
from .types import RolloutOutput


@dataclass(frozen=True)
class TrainBatchConfig:
    mini_batch_size: int
    micro_batch_size: int
    dynamic_batching: bool = False
    max_tokens_per_micro_batch: int | None = None
    packing_algorithm: Literal['ffd', 'kk'] = 'ffd'


def _extract_sampled_token_logps(logprobs: Any) -> list[float]:
    return [0.0 if not item else float(item[0][1]) for item in logprobs or []]


def sample_responses_to_rollout_rows(
    sources: list[dict[str, Any]],
    responses: list[SampleResponse],
    *,
    policy_version: int | None,
) -> list[RolloutOutput]:
    rows: list[RolloutOutput] = []
    for source, response in zip(sources, responses):
        for sequence in response.sequences:
            row = dict(source)
            row.update(sequence.new_input_feature or {})
            row['logprobs'] = _extract_sampled_token_logps(sequence.logprobs)
            row['stop_reason'] = sequence.stop_reason
            row['completion_length'] = len(sequence.tokens)
            row['rollout_policy_version'] = policy_version
            rows.append(row)
    return rows


def resolve_adapter_path(adapter_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(adapter_path)))
    if not os.path.exists(path):
        raise FileNotFoundError(f'local LoRA adapter path does not exist: {path}')
    return path


def sampler_data_parallel_size(sampler_gpus: int, sampler_tp: int) -> int:
    if sampler_gpus <= 0:
        raise ValueError(f'sampler_gpus must be positive, got {sampler_gpus}')
    if sampler_tp <= 0:
        raise ValueError(f'sampler_tp must be positive, got {sampler_tp}')
    if sampler_gpus % sampler_tp != 0:
        raise ValueError(f'sampler_gpus ({sampler_gpus}) must be divisible by sampler_tp ({sampler_tp})')
    return sampler_gpus // sampler_tp


def resolve_sequence_parallel_size(model_gpus: int, configured_size: int) -> int:
    if configured_size <= 0:
        raise ValueError(f'model.sequence_parallel_size must be positive, got {configured_size}')
    if model_gpus % configured_size:
        raise ValueError(f'runtime.model_gpus ({model_gpus}) must be divisible by '
                         f'model.sequence_parallel_size ({configured_size})')
    return configured_size


def resolve_model_attention_implementation(
    model_config: Mapping[str, Any],
    *,
    padding_free: bool,
    sequence_parallel_size: int,
) -> str | None:
    implementation = model_config.get('attn_implementation')
    if implementation is not None:
        implementation = str(implementation)
    if padding_free and sequence_parallel_size > 1 and implementation != 'flash_attention_2':
        raise ValueError('model.attn_implementation must be flash_attention_2 when '
                         'model.padding_free=true and model.sequence_parallel_size>1')
    return implementation


def build_native_fsdp_model_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    strategy = str(model_config.get('strategy', 'native_fsdp'))
    if strategy != 'native_fsdp':
        raise ValueError(f'model.strategy must be native_fsdp for RL training, got {strategy!r}')
    return {
        'strategy': strategy,
        'fsdp_config': dict(model_config.get('fsdp_config') or {}),
    }


def validate_context_batch_config(
    context_key: str,
    *,
    rollout_groups: int,
    num_generations: int,
    train: TrainBatchConfig,
    sampler_dp: int,
    model_dp: int,
) -> None:
    values = {
        'rollout.batch_size': rollout_groups,
        'rollout.num_generations': num_generations,
        'train.mini_batch_size': train.mini_batch_size,
        'train.micro_batch_size': train.micro_batch_size,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f'{name} for {context_key} must be positive, got {value}')
    if rollout_groups % sampler_dp:
        raise ValueError(f'rollout.batch_size for {context_key} must be divisible by sampler DP size '
                         f'({sampler_dp}), got {rollout_groups}')
    partition_samples = rollout_groups * num_generations
    if partition_samples % train.mini_batch_size:
        raise ValueError(f'partition for {context_key} has {partition_samples} samples and must be divisible by '
                         f'train.mini_batch_size={train.mini_batch_size}')
    if train.mini_batch_size % num_generations:
        raise ValueError(f'train.mini_batch_size for {context_key} must preserve complete prompt groups: '
                         f'{train.mini_batch_size} % {num_generations} != 0')
    if train.mini_batch_size % model_dp:
        raise ValueError(f'train.mini_batch_size for {context_key} must be divisible by '
                         f'model DP size {model_dp}')
    samples_per_rank = train.mini_batch_size // model_dp
    if train.micro_batch_size > samples_per_rank:
        raise ValueError(f'train.micro_batch_size for {context_key} must not exceed the per-rank train batch '
                         f'({samples_per_rank}), got {train.micro_batch_size}')
    if train.dynamic_batching:
        if train.max_tokens_per_micro_batch is None or train.max_tokens_per_micro_batch <= 0:
            raise ValueError(f'train.max_tokens_per_micro_batch for {context_key} must be positive when '
                             'train.dynamic_batching=true')
    if train.packing_algorithm not in ('ffd', 'kk'):
        raise ValueError(f'train.packing_algorithm for {context_key} must be ffd or kk, '
                         f'got {train.packing_algorithm!r}')


def configure_lora_lr_scheduler(
    model: Any,
    adapter_name: str,
    lora_config: Mapping[str, Any],
) -> None:
    scheduler_config = lora_config.get('lr_scheduler')
    if scheduler_config is None:
        return
    scheduler_config = dict(scheduler_config)
    scheduler_cls = scheduler_config.pop('cls')
    model.set_lr_scheduler(
        scheduler_cls,
        adapter_name=adapter_name,
        **scheduler_config,
    )


def resolve_context_learning_rate(
    train_config: Mapping[str, Any],
    lora_defaults: Mapping[str, Any],
) -> float:
    configured = train_config.get('learning_rate', lora_defaults.get('learning_rate'))
    if configured is None:
        raise ValueError('train.learning_rate or lora.learning_rate must be configured')
    learning_rate = float(configured)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(f'train.learning_rate must be a positive finite value, got {configured!r}')
    return learning_rate


def resolve_context_lora_target_modules(
    context_config: Mapping[str, Any],
    lora_defaults: Mapping[str, Any],
) -> str | list[str]:
    context_lora_config = dict(context_config.get('lora') or {})
    target_modules = context_lora_config.get(
        'target_modules',
        lora_defaults.get('target_modules', 'all-linear'),
    )
    if isinstance(target_modules, str):
        if not target_modules:
            raise ValueError('lora.target_modules must not be empty')
        return target_modules
    if isinstance(target_modules, Sequence) and target_modules:
        modules = list(target_modules)
        if all(isinstance(module, str) and module for module in modules):
            return modules
    raise ValueError('lora.target_modules must be a non-empty string or sequence of module names, '
                     f'got {target_modules!r}')


def resolve_context_loss_config(
    context_config: Mapping[str, Any],
    loss_defaults: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    loss_config: dict[str, Any] = {
        'cls': 'GRPOLoss',
        'epsilon': 0.2,
    }
    loss_config.update(dict(loss_defaults or {}))
    loss_config.update(dict(context_config.get('loss') or {}))

    loss_cls = loss_config.pop('cls', None)
    if not isinstance(loss_cls, str) or not loss_cls:
        raise ValueError(f'loss.cls must be a non-empty string, got {loss_cls!r}')
    return loss_cls, loss_config
