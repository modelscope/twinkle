# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Mapping


def _remap_names(values, name_mapping: Mapping[str, str]):
    """Remap parameter names while preserving numeric parameter IDs."""
    return [name_mapping.get(value, value) if isinstance(value, str) else value for value in values]


def remap_optimizer_state_names(state_dict: dict[str, Any], name_mapping: Mapping[str, str]) -> None:
    """Remap parameter identifiers in an optimizer state dict in place.

    Regular optimizers use numeric IDs, while full FSDP states use FQNs.
    Only string identifiers are remapped, so both formats share this path.
    """
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
