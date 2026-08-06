# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from torch import nn
from typing import Any, Dict, List, Optional, Union

from twinkle.utils import get_logger

logger = get_logger()

DEFAULT_TARGET_MODULES = ['attn', 'mlp']


@dataclass
class GaLoreConfig:
    """The configuration of the GaLore low-rank gradient projection.

    See https://arxiv.org/abs/2403.03507

    Args:
        rank: The galore rank.
        target_modules: The module name keys to enable galore on. If `None`, all `nn.Linear`/`nn.Embedding`
            modules whose name contains `attn` or `mlp` are used.
        update_proj_gap: The projection update interval for galore.
        galore_scale: The scale of the projected gradient.
        proj_type: The projection type, one of `std`, `reverse_std`, `right`, `left`, `full`.
    """
    rank: int = 128
    target_modules: Optional[Union[str, List[str]]] = None
    update_proj_gap: int = 50
    galore_scale: float = 1.0
    proj_type: str = 'std'

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = list(DEFAULT_TARGET_MODULES)
        elif isinstance(self.target_modules, str):
            self.target_modules = [self.target_modules]


def create_galore_param_groups(model: nn.Module, param_groups: List[Dict[str, Any]],
                               config: GaLoreConfig) -> List[Dict[str, Any]]:
    """Split the given param groups into galore groups and regular groups.

    The parameters matched by `config.target_modules` are moved into dedicated param groups carrying the
    galore keys (`rank`/`update_proj_gap`/`scale`/`proj_type`), which the GaLore optimizers detect in
    `step()`. Unmatched parameters stay in their original group, so their `weight_decay`/`lr` is preserved.

    Args:
        model: The model owning the parameters, used to resolve the galore target weights.
        param_groups: The param groups built by the caller, each with `params` and `param_names`.
        config: The galore config.
    Returns:
        The new param groups, ready to be passed to a GaLore optimizer.
    """
    galore_names = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if not any(target_key in module_name for target_key in config.target_modules):
            continue
        weight = getattr(module, 'weight', None)
        if weight is None or not weight.requires_grad:
            continue
        galore_names.add(f'{module_name}.weight')

    galore_defaults = {
        'rank': config.rank,
        'update_proj_gap': config.update_proj_gap,
        'scale': config.galore_scale,
        'proj_type': config.proj_type,
    }

    new_param_groups = []
    enabled_names = []
    for group in param_groups:
        names = group.get('param_names')
        assert names is not None, 'create_galore_param_groups requires `param_names` in every param group.'
        assert len(names) == len(group['params']), '`param_names` and `params` must have the same length.'

        matched_idx = [i for i, name in enumerate(names) if name in galore_names]
        rest_idx = [i for i, name in enumerate(names) if name not in galore_names]
        if matched_idx:
            galore_group = {key: value for key, value in group.items() if key not in ('params', 'param_names')}
            galore_group.update(galore_defaults)
            galore_group['params'] = [group['params'][i] for i in matched_idx]
            galore_group['param_names'] = [names[i] for i in matched_idx]
            new_param_groups.append(galore_group)
            enabled_names.extend(galore_group['param_names'])
        if rest_idx:
            rest_group = dict(group)
            rest_group['params'] = [group['params'][i] for i in rest_idx]
            rest_group['param_names'] = [names[i] for i in rest_idx]
            new_param_groups.append(rest_group)

    if not enabled_names:
        logger.warning(f'GaLore is enabled but no parameter matched target_modules={config.target_modules}, '
                       f'the optimizer will behave like a regular one.')
    else:
        logger.info(f'Enable GaLore for {len(enabled_names)} weights, e.g. {enabled_names[:3]}')
    return new_param_groups
