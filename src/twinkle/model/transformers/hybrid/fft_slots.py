# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, List

import torch
from peft.tuners.lora import LoraLayer
from peft.utils import ModulesToSaveWrapper
from torch import nn

from twinkle.model.multi_lora import MultiLora


class HybridFftSlots:
    """Manage the full-module FFT slots used by Hybrid adapters.

    ``MultiLora`` remains responsible for LoRA slots.  This class owns only
    the server-selected FFT modules and derives the matching ``fft_N`` slot
    from a tenant's existing LoRA slot index.
    """

    def __init__(self, multi_lora: MultiLora, s_fft: List[str]) -> None:
        """Bind the shared LoRA manager to its server-owned FFT allocation."""
        if not s_fft:
            raise ValueError('Hybrid requires at least one S_FFT module.')
        if len(set(s_fft)) != len(s_fft):
            raise ValueError('Hybrid S_FFT contains the same module more than once.')
        normalized_s_fft = [self._canonical_module_name(name) for name in s_fft]
        if len(set(normalized_s_fft)) != len(normalized_s_fft):
            raise ValueError('Hybrid S_FFT aliases resolve to the same layer.')
        self.multi_lora = multi_lora
        self.s_fft = normalized_s_fft
        self.hybrid_adapters: set[str] = set()
        self.allocated_to_layer_name: Dict[str, str] = {}
        self.allocated_to_wrapper_name: Dict[str, str] = {}

    @property
    def module(self):
        """Return the PEFT model that owns the FFT wrappers."""
        return self.multi_lora.module

    @staticmethod
    def _canonical_module_name(name: str) -> str:
        """Remove the module name prefix introduced by PEFT."""
        prefix = 'base_model.model.'
        return name[len(prefix):] if name.startswith(prefix) else name

    def install_fft_slots(self) -> None:
        """Install PEFT ``ModulesToSaveWrapper`` slots before DDP/FSDP wrapping."""
        if isinstance(self.module, list):
            raise NotImplementedError('Hybrid FFT slots currently require the Transformers backend.')
        named_modules = dict(self.module.named_modules())
        resolved_layer_names = set()
        for allocated_name in self.s_fft:
            matches = [
                (name, layer) for name, layer in named_modules.items()
                if self._canonical_module_name(name) == allocated_name
                and isinstance(layer, (LoraLayer, nn.Linear))
            ]
            if len(matches) != 1:
                raise ValueError(
                    f'Hybrid S_FFT module {allocated_name!r} resolved to {len(matches)} layers.')
            layer_name, layer = matches[0]
            if layer_name in resolved_layer_names:
                raise ValueError(
                    f'Hybrid S_FFT aliases resolve to the same layer {layer_name!r}.')
            resolved_layer_names.add(layer_name)

            if isinstance(layer, LoraLayer):
                original_module = layer.base_layer
                wrapper_name = f'{layer_name}.base_layer'
            else:
                original_module = layer
                wrapper_name = layer_name
            if any(parameter.is_meta for parameter in original_module.parameters()):
                raise ValueError('Hybrid FFT slots require materialized base weights.')
            wrapper = ModulesToSaveWrapper(original_module, 'fft_0')
            for slot in range(1, self.multi_lora.max_loras):
                wrapper.update(f'fft_{slot}')
            wrapper.set_adapter([])
            for parameter in wrapper.modules_to_save.parameters():
                parameter.requires_grad_(True)
            if isinstance(layer, LoraLayer):
                layer.base_layer = wrapper
            else:
                parent_name, _, child_name = layer_name.rpartition('.')
                parent = self.module.get_submodule(parent_name) if parent_name else self.module
                setattr(parent, child_name, wrapper)
            self.allocated_to_layer_name[allocated_name] = layer_name
            self.allocated_to_wrapper_name[allocated_name] = wrapper_name

    def is_hybrid(self, adapter_name: str) -> bool:
        """Return whether an adapter owns an FFT slot."""
        return adapter_name in self.hybrid_adapters

    def register_adapter(self, adapter_name: str) -> None:
        """Mark an existing LoRA tenant as a Hybrid tenant."""
        self.hybrid_adapters.add(adapter_name)

    def unregister_adapter(self, adapter_name: str) -> None:
        """Remove Hybrid ownership before releasing a tenant slot."""
        self.hybrid_adapters.discard(adapter_name)

    def _tenant(self, adapter_name: str):
        """Look up the LoRA tenant that determines the FFT slot index."""
        return self.multi_lora.find_lora_by_tenant(adapter_name)

    def _fft_adapter_name(self, adapter_name: str) -> str:
        """Map a tenant's LoRA slot index to its PEFT FFT adapter name."""
        return f'fft_{self._tenant(adapter_name).index}'

    def _get_fft_wrapper(self, allocated_name: str) -> ModulesToSaveWrapper:
        """Resolve one allocated module to its installed PEFT wrapper."""
        return self.module.get_submodule(self.allocated_to_wrapper_name[allocated_name])

    def _iter_fft_wrappers(self):
        """Return all wrappers in stable allocation order."""
        return [self._get_fft_wrapper(name) for name in self.s_fft]

    def activate_fft_slot(self, adapter_name: str) -> None:
        """Activate this tenant's FFT copies, or disable FFT for regular LoRA."""
        fft_adapter_name = self._fft_adapter_name(adapter_name) if self.is_hybrid(adapter_name) else None
        for wrapper in self._iter_fft_wrappers():
            wrapper.set_adapter(fft_adapter_name if fft_adapter_name is not None else [])

    def deactivate_fft_slots(self) -> None:
        """Disable every FFT wrapper after a model operation."""
        for wrapper in self._iter_fft_wrappers():
            wrapper.set_adapter([])

    def resolve_lora_targets(self, target_modules) -> List[str]:
        """Resolve tenant LoRA targets while reserving S_FFT for full tuning."""
        lora_layers = self.multi_lora.lora_layer_names
        fft_layers = set(self.allocated_to_layer_name.values())
        if target_modules is None:
            selected = lora_layers
        else:
            selected = [
                name for name in lora_layers
                if self.multi_lora.match_target_modules(name, target_modules)
            ]
        return sorted(name for name in selected if name not in fft_layers)

    def _tenant_lora_layer_names(self, adapter_name: str) -> List[str]:
        """Return this Hybrid tenant's LoRA layers, excluding its FFT layers."""
        tenant = self._tenant(adapter_name)
        fft_layers = set(self.allocated_to_layer_name.values())
        return [
            name for name in self.multi_lora.lora_layer_names
            if name not in fft_layers
            and self.multi_lora.match_target_modules(name, tenant.tenant_config.target_modules)
        ]

    def _iter_fft_slot_parameters(self, adapter_name: str):
        """Yield all parameters belonging to one tenant's FFT module copies."""
        fft_adapter_name = self._fft_adapter_name(adapter_name)
        for allocated_name in self.s_fft:
            wrapper_name = self.allocated_to_wrapper_name[allocated_name]
            slot_module = self._get_fft_wrapper(allocated_name).modules_to_save[fft_adapter_name]
            for parameter_name, parameter in slot_module.named_parameters():
                yield allocated_name, wrapper_name, fft_adapter_name, parameter_name, parameter

    def reset_adapter_slot(self, adapter_name: str) -> None:
        """Restore this tenant's FFT copies from the frozen original modules."""
        for allocated_name, _, _, parameter_name, target in self._iter_fft_slot_parameters(adapter_name):
            wrapper = self._get_fft_wrapper(allocated_name)
            original_parameters = dict(wrapper.original_module.named_parameters())
            self.multi_lora._write_param_tensor(
                target, self.multi_lora._read_param_tensor(original_parameters[parameter_name]))

    @staticmethod
    def _checkpoint_key(allocated_name: str, parameter_name: str) -> str:
        """Build the plain Transformers checkpoint key for an FFT tensor."""
        return f'base_model.model.{allocated_name}.{parameter_name}'

    def get_fft_state_dict(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        """Return an independent snapshot of one tenant's FFT state."""
        if not self.is_hybrid(adapter_name):
            return {}
        state = {}
        for allocated_name, _, _, parameter_name, parameter in self._iter_fft_slot_parameters(adapter_name):
            state[self._checkpoint_key(allocated_name, parameter_name)] = (
                self.multi_lora._read_param_tensor(parameter).detach().clone())
        return state

    def set_fft_state_dict(self, adapter_name: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore one tenant's FFT copies from lossless training state."""
        if not self.is_hybrid(adapter_name):
            return
        for allocated_name, _, _, parameter_name, parameter in self._iter_fft_slot_parameters(adapter_name):
            key = self._checkpoint_key(allocated_name, parameter_name)
            if key not in state_dict:
                raise ValueError(f'Hybrid training state is missing {key!r}.')
            self.multi_lora._write_param_tensor(parameter, state_dict[key])

    def named_fft_parameters(self, adapter_name: str):
        """Return trainable FFT parameters with their PEFT state-dict names."""
        if not self.is_hybrid(adapter_name):
            return []
        return [
            (f'{wrapper_name}.modules_to_save.{fft_adapter_name}.{parameter_name}', parameter)
            for _, wrapper_name, fft_adapter_name, parameter_name, parameter
            in self._iter_fft_slot_parameters(adapter_name)
        ]

    @staticmethod
    def _normalize_base_state_key(name: str) -> str:
        """Convert a PEFT base-module key to its plain Transformers key."""
        prefix = 'base_model.model.'
        if name.startswith(prefix):
            name = name[len(prefix):]
        name = name.replace('.base_layer.original_module.', '.')
        name = name.replace('.original_module.', '.')
        return name.replace('.base_layer.', '.')

    def iter_merged_state_dict(self, adapter_name: str, full_state_dict: Dict[str, torch.Tensor]):
        """Yield a non-destructively merged plain Transformers state dict."""
        if not self.is_hybrid(adapter_name):
            raise ValueError(f'Adapter {adapter_name!r} is not a Hybrid adapter.')
        tenant = self._tenant(adapter_name)
        slot = tenant.adapter_name
        replacements: Dict[str, torch.Tensor] = {}
        for layer_name in self._tenant_lora_layer_names(adapter_name):
            base_key = f'{layer_name}.base_layer.weight'
            a_key = f'{layer_name}.lora_A.{slot}.weight'
            b_key = f'{layer_name}.lora_B.{slot}.weight'
            missing = [key for key in (base_key, a_key, b_key) if key not in full_state_dict]
            if missing:
                raise ValueError(f'Cannot export Hybrid module {layer_name!r}; missing {missing}.')
            base = full_state_dict[base_key]
            rank = tenant.tenant_config.r
            a = full_state_dict[a_key][:rank, :]
            b = full_state_dict[b_key][:, :rank]
            scaling = tenant.tenant_config.lora_alpha / (
                rank**0.5 if tenant.tenant_config.use_rslora else rank)
            delta = b.to(torch.float32) @ a.to(torch.float32)
            if getattr(tenant.tenant_config, 'fan_in_fan_out', False):
                delta = delta.transpose(0, 1)
            replacements[base_key] = base + delta.to(dtype=base.dtype) * scaling

        for allocated_name, wrapper_name, fft_adapter_name, parameter_name, _ in self._iter_fft_slot_parameters(
                adapter_name):
            base_key = f'{wrapper_name}.original_module.{parameter_name}'
            fft_key = f'{wrapper_name}.modules_to_save.{fft_adapter_name}.{parameter_name}'
            if base_key not in full_state_dict or fft_key not in full_state_dict:
                raise ValueError(f'Cannot export Hybrid FFT layer {allocated_name!r}.')
            replacements[base_key] = full_state_dict[fft_key]

        emitted = set()
        for name, value in full_state_dict.items():
            if ('.lora_' in name or '.modules_to_save.' in name
                    or self.multi_lora._is_target_parameter_lora_name(name)):
                continue
            output_name = self._normalize_base_state_key(name)
            if output_name in emitted:
                continue
            emitted.add(output_name)
            yield output_name, replacements.get(name, value).detach().cpu()

    def build_merged_state_dict(self, adapter_name: str, full_state_dict: Dict[str, torch.Tensor]):
        """Materialize the merged state dict for deployment-oriented saving."""
        return dict(self.iter_merged_state_dict(adapter_name, full_state_dict))

    def build_training_state_dict(self, adapter_name: str, full_state_dict: Dict[str, torch.Tensor]):
        """Extract lossless LoRA and FFT state for a Hybrid tenant."""
        if not self.is_hybrid(adapter_name):
            raise ValueError(f'Adapter {adapter_name!r} is not a Hybrid adapter.')
        tenant = self._tenant(adapter_name)
        state = {}
        for layer_name in self._tenant_lora_layer_names(adapter_name):
            for kind in ('A', 'B'):
                source = f'{layer_name}.lora_{kind}.{tenant.adapter_name}.weight'
                if source not in full_state_dict:
                    raise ValueError(f'Hybrid training state is missing {source!r}.')
                value = self.multi_lora._slice_rank_tensor(
                    source, full_state_dict[source], tenant.tenant_config.r)
                state[source.replace(f'.{tenant.adapter_name}.', '.')] = value.detach().cpu()
        for allocated_name, wrapper_name, fft_adapter_name, parameter_name, _ in self._iter_fft_slot_parameters(
                adapter_name):
            source = f'{wrapper_name}.modules_to_save.{fft_adapter_name}.{parameter_name}'
            if source not in full_state_dict:
                raise ValueError(f'Hybrid training state is missing {source!r}.')
            state[self._checkpoint_key(allocated_name, parameter_name)] = full_state_dict[source].detach().cpu()
        return state
