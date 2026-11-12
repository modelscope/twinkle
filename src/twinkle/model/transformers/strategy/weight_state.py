# Copyright (c) ModelScope Contributors. All rights reserved.


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
