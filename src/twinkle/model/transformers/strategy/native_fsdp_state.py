# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Mapping

LORA_STATE_KEY_MARKERS = ('lora_A', 'lora_B', 'lora_embedding')
PEFT_BASE_PREFIX = 'base_model.model.'
PEFT_BASE_LAYER_SEGMENT = 'base_layer'


def _is_lora_state_key(name: str) -> bool:
    return any(marker in name for marker in LORA_STATE_KEY_MARKERS)


def _strip_peft_base_prefix(name: str) -> str:
    while name.startswith(PEFT_BASE_PREFIX):
        name = name[len(PEFT_BASE_PREFIX):]
    return name


def _strip_base_layer_segments(name: str) -> str:
    return '.'.join(segment for segment in name.split('.') if segment != PEFT_BASE_LAYER_SEGMENT)


def _source_key_candidates(param_name: str) -> list[str]:
    stripped_prefix = _strip_peft_base_prefix(param_name)
    candidates = [
        param_name,
        stripped_prefix,
        _strip_base_layer_segments(param_name),
        _strip_base_layer_segments(stripped_prefix),
    ]
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resolve_full_state_source_key(param_name: str, source_state: Mapping[str, Any]) -> str:
    if _is_lora_state_key(param_name):
        raise KeyError(f"LoRA parameter '{param_name}' must be loaded from adapter source state.")

    candidates = _source_key_candidates(param_name)
    for candidate in candidates:
        if candidate in source_state:
            return candidate
    raise KeyError(
        f"Missing source metadata for parameter '{param_name}'. "
        f'Tried source keys: {", ".join(candidates)}.')


def _collect_adapter_source_state(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    adapter_state = {}
    for name, tensor in state_dict.items():
        if not _is_lora_state_key(name) or not hasattr(tensor, 'detach'):
            continue
        if getattr(tensor, 'is_meta', False):
            continue
        adapter_state[name] = tensor.detach().cpu().clone()
    return adapter_state


def _collect_state_metadata(state_dict: Mapping[str, Any]) -> Dict[str, tuple[tuple[int, ...], Any]]:
    return {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in state_dict.items()
        if hasattr(tensor, 'shape') and hasattr(tensor, 'dtype')
    }
