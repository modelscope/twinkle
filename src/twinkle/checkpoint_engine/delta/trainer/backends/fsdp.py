# Copyright (c) ModelScope Contributors. All rights reserved.
"""FSDP2 implementation of the delta trainer backend."""
from __future__ import annotations

import re

import torch

from twinkle import Platform
from ..spec import ShardSpec, derive_dtensor_placement
from .base import DeltaTrainerBackend, TensorShards, TensorWeights


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model):
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


def validate_fsdp2(owner) -> None:
    from torch.distributed.tensor import DTensor
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

    platform_name = Platform.get_platform().__name__
    if platform_name not in ('GPU', 'NPU'):
        raise ValueError(f'delta_sharded requires GPU or NPU trainer actors, got platform {platform_name!r}')
    if not isinstance(owner.strategy, (NativeFSDPStrategy, AccelerateStrategy)):
        raise ValueError('delta_sharded requires native or Accelerate FSDP2')
    model = owner.strategy.unwrap_model(owner.model)
    if getattr(model, 'peft_config', None) or getattr(model.config, 'quantization_config', None):
        raise ValueError('delta_sharded does not support adapters or quantization')
    config = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
    if any(getattr(config, name, 0) for name in ('num_experts', 'moe_num_experts',
                                               'n_routed_experts', 'num_local_experts')):
        raise ValueError('Twinkle FSDP delta export does not yet implement backend-specific MoE HF conversion')
    if any((getattr(owner.device_mesh, dim, 1) or 1) != 1
           for dim in ('tp_world_size', 'pp_world_size', 'ep_world_size')):
        raise ValueError('FSDP delta export requires training TP=PP=EP=1')
    state = model.state_dict()
    # FSDP leaves the model unwrapped on a single rank; its plain tensors export as unsharded parameters.
    single_rank = getattr(owner.device_mesh, 'world_size', None) == 1
    if not single_rank and not any(isinstance(param, DTensor) for param in state.values()):
        raise ValueError('delta_sharded requires a wrapped FSDP2 DTensor model')
    if any(param.numel() >= 1 << 31 for param in state.values()):
        raise ValueError('delta_sharded indices require fewer than 2^31 elements per tensor')
    if any(not param.is_floating_point() for param in state.values()):
        raise ValueError('FSDP delta checkpoint tensors must be floating point')
    for param in state.values():
        derive_dtensor_placement(ShardSpec.from_param(param))


class FSDP2DeltaBackend(DeltaTrainerBackend):
    """Delta backend for native/Accelerate FSDP2 models (identity HF coordinates)."""

    def validate_delta_backend(self) -> None:
        validate_fsdp2(self)

    def _delta_state_dict(self) -> dict[str, torch.Tensor]:
        model = self.strategy.unwrap_model(self.model)
        return convert_weight_keys(model.state_dict(), model)

    def get_per_tensor_param(self) -> tuple[TensorWeights, None]:
        def generate():
            from torch.distributed.tensor import DTensor

            for name, param in self._delta_state_dict().items():
                param = param.to(self._delta_device, non_blocking=True)
                if isinstance(param, DTensor):
                    param = param.full_tensor()
                yield name, param
        return generate(), None

    def get_per_tensor_param_shard(self) -> tuple[TensorShards, None]:
        def generate():
            from torch.distributed.tensor import DTensor

            for name, param in self._delta_state_dict().items():
                spec = ShardSpec.from_param(param)
                param = param.to(self._delta_device, non_blocking=True)
                if param.is_floating_point():
                    param = param.to(torch.bfloat16, non_blocking=True)
                local = param.to_local() if isinstance(param, DTensor) else param
                yield name, local.reshape(-1), spec
        return generate(), None
