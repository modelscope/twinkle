from __future__ import annotations

import math
import torch
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from peft import LoraConfig
from torch import nn
from typing import Iterable, Iterator


class LoraParameterProxy(nn.Module):

    def __init__(self, LoraWrapper, slot_name):
        super().__init__()
        self.LoraWrapper = LoraWrapper
        self.slot_name = slot_name

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta_weight = self.LoraWrapper.get_delta_weight(self.slot_name)
        mix_weight = weight + delta_weight
        return mix_weight


@dataclass
class TargetParameterRecord:
    module_name: str
    module: nn.Module
    parameter_name: str

    @property
    def key(self) -> str:
        if self.module_name:
            return f'{self.module_name}.{self.parameter_name}'
        return self.parameter_name


class TargetParameterLoraWrapper(nn.Module):

    def __init__(self, record: TargetParameterRecord, max_loras: int, max_r: int):
        super().__init__()
        self.record = record
        # Unsharded original target parameter (pre-sharding snapshot)
        parameter = getattr(self.record.module, self.record.parameter_name)
        self.original_parameter_ndim = parameter.ndim
        # Cache invariant attributes that won't change after initialization
        self.did_swap_in_out_features = parameter.ndim == 3 and not getattr(self.record.module, 'is_transposed', False)
        if parameter.ndim == 3:
            self.in_features = parameter.shape[-1] if self.did_swap_in_out_features else parameter.shape[-2]
            self.out_features = parameter.shape[-2] if self.did_swap_in_out_features else parameter.shape[-1]
        else:
            self.out_features, self.in_features = parameter.shape[0], parameter.shape[1]

        self.max_loras = max_loras
        self.max_r = max_r
        self.active_adapter: str | None = None
        self.disable_adapters = False
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scaling: dict[str, float] = {}
        self.r: dict[str, int] = {}
        self._initial_lora_A: dict[str, torch.Tensor] = {}
        self.peft_key_prefix = ''
        self._init_slots()

    @property
    def base_parameter(self) -> nn.Parameter:
        module = self.record.module
        param_name = self.record.parameter_name
        if hasattr(module, 'parametrizations'):
            if hasattr(module.parametrizations, param_name):
                return module.parametrizations[param_name].original
        return getattr(module, param_name)

    @property
    def num_experts(self) -> int:
        # Check if the module has EP sharding info (for tensor experts)
        if hasattr(self.record.module, '_ep_local_end') and hasattr(self.record.module, '_ep_local_start'):
            return self.record.module._ep_local_end - self.record.module._ep_local_start  # 在wrap_model的ep切分后会设置
        # Check if the module has EP sharding info (for ModuleList experts)
        if hasattr(self.record.module, '_ep_experts_per_rank'):
            return self.record.module._ep_experts_per_rank
        # Fallback to base parameter shape
        parameter = self.base_parameter
        return parameter.shape[0] if parameter.ndim == 3 else 1

    def _init_slots(self) -> None:
        parameter = self.base_parameter
        if parameter.ndim not in (2, 3):
            raise ValueError(
                f'target parameter {self.record.key} has {parameter.ndim} dimensions; only 2D and 3D are supported')

        # Note: reset_slot requires the tensor to be created on a physical device, not on a meta device.
        device = parameter.device
        if device.type == 'meta':
            device = 'cpu'
        for index in range(self.max_loras):
            slot_name = f'lora_{index}'
            self.lora_A[slot_name] = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    self.max_r,
                    self.in_features,
                    device=device,
                    dtype=parameter.dtype,
                ))
            self.lora_B[slot_name] = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    self.out_features,
                    self.max_r,
                    device=device,
                    dtype=parameter.dtype,
                ))
            self.r[slot_name] = self.max_r
            self.scaling[slot_name] = 1.0
            self.reset_slot(slot_name)

    def reset_slot(self, slot_name: str) -> None:
        if slot_name not in self._initial_lora_A:
            nn.init.kaiming_uniform_(self.lora_A[slot_name], a=math.sqrt(5))
            self._initial_lora_A[slot_name] = self.lora_A[slot_name].detach().clone().cpu()
        else:
            initial = self._initial_lora_A[slot_name]
            if hasattr(self.record.module, '_ep_local_start') and hasattr(self.record.module, '_ep_local_end'):
                start = self.record.module._ep_local_start
                end = self.record.module._ep_local_end
                initial = initial[start:end]
            initial = initial.to(
                device=self.lora_A[slot_name].device,
                dtype=self.lora_A[slot_name].dtype,
            )
            self.lora_A[slot_name].data.copy_(initial)
        nn.init.zeros_(self.lora_B[slot_name])

    def configure_slot(self, slot_name: str, config: LoraConfig) -> None:
        if slot_name not in self.lora_A:
            raise ValueError(f'Unknown target-parameter LoRA slot: {slot_name}')
        if config.r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {config.r}')
        if config.r > self.max_r:
            raise ValueError(f'LoRA rank {config.r} exceeds max_r={self.max_r}')
        if getattr(config, 'lora_dropout', 0):
            raise ValueError('target_parameters LoRA does not support lora_dropout != 0')
        if getattr(config, 'fan_in_fan_out', False):
            raise ValueError('target_parameters LoRA does not support fan_in_fan_out=True')
        if getattr(config, 'use_dora', False):
            raise ValueError('target_parameters LoRA does not support use_dora=True')
        if getattr(config, 'lora_bias', False):
            raise ValueError('target_parameters LoRA does not support lora_bias=True')

        self.r[slot_name] = config.r
        if getattr(config, 'use_rslora', False):
            self.scaling[slot_name] = config.lora_alpha / math.sqrt(config.r)
        else:
            self.scaling[slot_name] = config.lora_alpha / config.r

    def get_delta_weight(self, slot_name: str) -> torch.Tensor:
        if slot_name not in self.lora_A:
            raise ValueError(f'Unknown target-parameter LoRA slot: {slot_name}')

        r = self.r[slot_name]
        num_experts = self.num_experts
        if hasattr(self.record.module, '_ep_local_start'):
            # EP sharded: use full sharded weights
            weight_A = self.lora_A[slot_name][:, :r, :]
            weight_B = self.lora_B[slot_name][:, :, :r]
        else:
            # Not sharded: slice to actual rank
            weight_A = self.lora_A[slot_name][:num_experts, :r, :]
            weight_B = self.lora_B[slot_name][:num_experts, :, :r]

        # Don't call base_parameter during LoRA injection (infinite recursion risk)
        if self.original_parameter_ndim == 2:
            return (weight_B[0] @ weight_A[0]) * self.scaling[slot_name]

        if self.did_swap_in_out_features:
            return torch.einsum('e o r, e r i -> e o i', weight_B, weight_A) * self.scaling[slot_name]
        return torch.einsum('e o r, e r i -> e i o', weight_B, weight_A) * self.scaling[slot_name]

    @contextmanager
    def activate(self, slot_name: str | None, disable_lora: bool = False):
        if disable_lora or slot_name is None or slot_name not in self.lora_A:
            yield
            return

        module = self.record.module
        param_name = self.record.parameter_name
        already_parametrized = nn.utils.parametrize.is_parametrized(module, param_name)
        if not already_parametrized:
            # LoRA weights change after EP + FSDP sharding, so they must be computed dynamically and not be pre-fixed.
            # delta_weight = self.get_delta_weight(slot_name) # lora_weight = B @ A * scaling
            requires_grad_before = self.base_parameter.requires_grad
            nn.utils.parametrize.register_parametrization(
                self.record.module,
                self.record.parameter_name,
                LoraParameterProxy(self, slot_name),
            )
            module.parametrizations[param_name].original.requires_grad_(requires_grad_before)
        try:
            with nn.utils.parametrize.cached():
                yield
        finally:
            if not already_parametrized:
                nn.utils.parametrize.remove_parametrizations(
                    self.record.module,
                    self.record.parameter_name,
                    leave_parametrized=False,
                )

    def named_slot_parameters(self, slot_name: str) -> Iterator[tuple[str, nn.Parameter]]:
        if slot_name not in self.lora_A:
            return
        yield f'{self.record.key}.lora_A.{slot_name}.weight', self.lora_A[slot_name]
        yield f'{self.record.key}.lora_B.{slot_name}.weight', self.lora_B[slot_name]

    def parameters_for_slot(self, slot_name: str) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_slot_parameters(slot_name)]

    def get_state_dict(self, slot_name: str) -> dict[str, torch.Tensor]:
        r = self.r[slot_name]
        num_experts = self.num_experts
        # Slice to actual rank first, keep 3D shape: [num_experts, r, in_features]
        lora_A = self.lora_A[slot_name][:num_experts, :r, :].detach().clone()
        # Keep 3D shape: [num_experts, out_features, r]
        lora_B = self.lora_B[slot_name][:num_experts, :, :r].detach().clone()
        return {
            f'{self.peft_key_prefix}.lora_A.weight': lora_A,
            f'{self.peft_key_prefix}.lora_B.weight': lora_B,
        }

    def set_state_dict(self, slot_name: str, state_dict: dict[str, torch.Tensor]) -> set[str]:
        key_a = f'{self.peft_key_prefix}.lora_A.weight'
        key_b = f'{self.peft_key_prefix}.lora_B.weight'
        if key_a not in state_dict and key_b not in state_dict:
            return set()
        if key_a not in state_dict or key_b not in state_dict:
            raise KeyError(f'Missing target-parameter LoRA pair for {self.peft_key_prefix}')

        r = self.r[slot_name]
        num_experts = self.num_experts

        with torch.no_grad():
            self.lora_A[slot_name].zero_()
            self.lora_B[slot_name].zero_()
            # Directly use 3D tensor: [num_experts, r, in_features]
            lora_A_3d = state_dict[key_a].to(
                device=self.lora_A[slot_name].device,
                dtype=self.lora_A[slot_name].dtype,
            )
            # Validate shape
            expected_shape_a = (num_experts, r, self.lora_A[slot_name].shape[-1])
            if lora_A_3d.shape != expected_shape_a:
                raise ValueError(f'Expected lora_A shape {expected_shape_a}, got {lora_A_3d.shape}')
            self.lora_A[slot_name][:num_experts, :r, :].copy_(lora_A_3d)

            # Directly use 3D tensor: [num_experts, out_features, r]
            lora_B_3d = state_dict[key_b].to(
                device=self.lora_B[slot_name].device,
                dtype=self.lora_B[slot_name].dtype,
            )
            # Validate shape
            expected_shape_b = (num_experts, self.lora_B[slot_name].shape[1], r)
            if lora_B_3d.shape != expected_shape_b:
                raise ValueError(f'Expected lora_B shape {expected_shape_b}, got {lora_B_3d.shape}')
            self.lora_B[slot_name][:num_experts, :, :r].copy_(lora_B_3d)

        return {key_a, key_b}


class TargetParameterLoraManager:

    def __init__(self, max_loras: int, max_r: int):
        self.max_loras = max_loras
        self.max_r = max_r
        self.wrappers: list[TargetParameterLoraWrapper] = []
        self.tenant_to_slot: dict[str, str] = {}
        self.tenant_configs: dict[str, LoraConfig] = {}
        self._target_parameters: tuple[str, ...] | None = None

    def patch(self, model: nn.Module, target_parameters: Iterable[str]) -> None:
        target_parameters = tuple(target_parameters)
        if not target_parameters:
            return
        if self._target_parameters is not None:
            if self._target_parameters != target_parameters:
                raise ValueError(
                    f'target_parameters already patched as {self._target_parameters}, got {target_parameters}')
            return

        records = []
        for module_name, module in model.named_modules():
            for param_name, parameter in module.named_parameters(recurse=False):
                key = f'{module_name}.{param_name}' if module_name else param_name
                if key in target_parameters or any(key.endswith(f'.{target}') for target in target_parameters):
                    if parameter.ndim not in (2, 3):
                        raise ValueError(
                            f'target parameter {key} has {parameter.ndim} dimensions; only 2D and 3D are supported')
                    records.append(TargetParameterRecord(module_name, module, param_name))

        if not records:
            raise ValueError(f'target_parameters={target_parameters} were set but no parameter was matched')

        for record in records:
            wrapper = TargetParameterLoraWrapper(record, max_loras=self.max_loras, max_r=self.max_r)
            record.module.add_module(f'_twinkle_lora_{record.parameter_name}', wrapper)
            self.wrappers.append(wrapper)
        self._assign_peft_key_prefixes()
        self._target_parameters = target_parameters

    def _assign_peft_key_prefixes(self) -> None:
        wrappers_by_module: dict[str, list[TargetParameterLoraWrapper]] = {}
        for wrapper in self.wrappers:
            wrappers_by_module.setdefault(wrapper.record.module_name, []).append(wrapper)

        for module_name, wrappers in wrappers_by_module.items():
            module_prefix = f'base_model.model.{module_name}' if module_name else 'base_model.model'
            for index, wrapper in enumerate(wrappers):
                nested_prefix = '.'.join(['base_layer'] * (len(wrappers) - index - 1))
                wrapper.peft_key_prefix = f'{module_prefix}.{nested_prefix}' if nested_prefix else module_prefix

    def acquire(self, tenant_adapter_name: str, slot_name: str, config: LoraConfig) -> None:
        if tenant_adapter_name in self.tenant_to_slot:
            raise ValueError(f'Lora {tenant_adapter_name} already exists')
        if getattr(config, 'target_parameters', None) and not self.wrappers:
            raise ValueError('target_parameters LoRA slots must be patched before acquire')

        self.tenant_to_slot[tenant_adapter_name] = slot_name
        self.tenant_configs[tenant_adapter_name] = config
        for wrapper in self.wrappers:
            wrapper.configure_slot(slot_name, config)

    def release(self, tenant_adapter_name: str) -> None:
        slot_name = self.tenant_to_slot.pop(tenant_adapter_name, None)
        self.tenant_configs.pop(tenant_adapter_name, None)
        if slot_name is None:
            return
        for wrapper in self.wrappers:
            wrapper.reset_slot(slot_name)

    @contextmanager
    def adapter(self, tenant_adapter_name: str, disable_lora: bool = False):
        slot_name = self.tenant_to_slot.get(tenant_adapter_name)
        with ExitStack() as stack:
            for wrapper in self.wrappers:
                stack.enter_context(wrapper.activate(slot_name, disable_lora=disable_lora))
            yield

    def parameters_for_tenant(self, tenant_adapter_name: str) -> list[nn.Parameter]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        parameters = []
        for wrapper in self.wrappers:
            parameters.extend(wrapper.parameters_for_slot(slot_name))
        return parameters

    def named_slot_parameters(self, tenant_adapter_name: str) -> Iterator[tuple[str, nn.Parameter]]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        for wrapper in self.wrappers:
            yield from wrapper.named_slot_parameters(slot_name)

    def get_state_dict(self, tenant_adapter_name: str) -> dict[str, torch.Tensor]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        state_dict = {}
        for wrapper in self.wrappers:
            state_dict.update(wrapper.get_state_dict(slot_name))
        return state_dict

    def set_state_dict(self, tenant_adapter_name: str, state_dict: dict[str, torch.Tensor]) -> set[str]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        consumed_keys = set()
        for wrapper in self.wrappers:
            consumed_keys.update(wrapper.set_state_dict(slot_name, state_dict))
        return consumed_keys
