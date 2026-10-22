# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint state-dict helpers shared by the transformers strategies and the model layer.

The strategies own *collection* (how sharded tensors reach host memory), this module owns
*key semantics* (which names belong to an adapter, how they normalize for checkpoints, and
how identifiers remap when a checkpoint crosses name conventions):

- ``is_lora_state_key`` / ``is_saveable_adapter_key`` -- the single source of truth for
  adapter-key filtering. The strategies previously each carried a private copy, and the
  deepspeed copy forgot the ``modules_to_save`` segment, silently dropping fully trained
  modules from adapter checkpoints.
- ``collect_local_state`` -- the single collection primitive. It enumerates parameters and,
  on request, persistent buffers. PEFT ``modules_to_save`` trains a full copy of a module,
  whose persistent buffers (calibration statistics, temperature, ...) must ride along in
  every adapter/full checkpoint; enumerating ``named_parameters()`` alone loses them.
- ``normalize_adapter_keys`` -- strips the PEFT wrapper segments under which checkpoint
  consumers expect the keys.
- ``remap_optimizer_state_names`` -- renames the parameter identifiers inside an optimizer
  state dict (regular optimizers use numeric IDs, full FSDP states use FQNs; only string
  identifiers are remapped, so both formats share this path).
- ``with_tied_weight_aliases`` -- re-expands the checkpoint names omitted for parameters
  shared by identity (tied weights).
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

LORA_STATE_KEY_MARKERS = ('lora_A', 'lora_B', 'lora_embedding')
#: PEFT wraps a fully-trained module copy as ``<parent>.modules_to_save.<adapter>.<module>``.
MODULES_TO_SAVE_SEGMENT = '.modules_to_save.'


def is_lora_state_key(name: str) -> bool:
    return any(marker in name for marker in LORA_STATE_KEY_MARKERS)


def is_saveable_adapter_key(name: str) -> bool:
    """LoRA deltas plus the full trainable copies under ``modules_to_save``."""
    return is_lora_state_key(name) or MODULES_TO_SAVE_SEGMENT in name


def collect_local_state(module: torch.nn.Module, *, include_buffers: bool = False) -> Dict[str, torch.Tensor]:
    """Return the module's parameters (and optionally persistent buffers) by qualified name.

    The values are the live tensors on whatever device the strategy keeps them; converting
    to a gathered/cpu copy stays with the caller (sharding details differ per strategy).
    """
    state: Dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        state[name] = param
    if include_buffers:
        for module_name, submodule in module.named_modules():
            for buffer_name, buffer in submodule.named_buffers(recurse=False):
                if buffer_name in submodule._non_persistent_buffers_set:
                    continue
                key = f'{module_name}.{buffer_name}' if module_name else buffer_name
                state[key] = buffer
    return state


def normalize_adapter_keys(state: Dict[str, Any], adapter_name: str) -> Dict[str, Any]:
    """Strip the PEFT wrapper segments adapter checkpoints store their keys without.

    ``...modules_to_save.<adapter>.x`` collapses to ``...x`` (the wrapper is transparent
    on load), other ``...<adapter>.x`` keys collapse their adapter segment likewise.
    """
    modules_to_save_infix = f'{MODULES_TO_SAVE_SEGMENT}{adapter_name}.'
    adapter_suffix = f'.{adapter_name}.'
    processed = {}
    for key, value in state.items():
        if modules_to_save_infix in key:
            normalized = key.replace(modules_to_save_infix, '.')
        else:
            normalized = key.replace(adapter_suffix, '.')
        processed[normalized] = value
    return processed


def _remap_names(values, name_mapping: Mapping[str, str]):
    """Remap parameter names while preserving numeric parameter IDs."""
    return [name_mapping.get(value, value) if isinstance(value, str) else value for value in values]


def remap_optimizer_state_names(state_dict: dict[str, Any], name_mapping: Mapping[str, str]) -> None:
    """Remap parameter identifiers in an optimizer state dict in place."""
    if not name_mapping:
        return

    state = state_dict.get('state')
    if isinstance(state, dict):
        state_dict['state'] = {
            name_mapping.get(name, name) if isinstance(name, str) else name: value
            for name, value in state.items()
        }

    for group in state_dict.get('param_groups', []):
        for field in ('params', 'param_names'):
            values = group.get(field)
            if isinstance(values, (list, tuple)):
                group[field] = _remap_names(values, name_mapping)


def with_tied_weight_aliases(model, state_dict: dict) -> dict:
    """Expand omitted checkpoint names only for parameters shared by identity."""
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    expanded = dict(state_dict)
    for names in aliases.values():
        if len(names) < 2:
            continue
        source = next((name for name in names if name in state_dict), None)
        if source is not None:
            for name in names:
                expanded.setdefault(name, state_dict[source])
    return expanded
