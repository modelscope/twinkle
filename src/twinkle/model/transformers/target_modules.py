# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model-agnostic multimodal LoRA target expansion for the transformers backend.

When a LoRA run targets ``'all-linear'`` on a multimodal model, "all linear layers" has to respect the
``freeze_llm`` / ``freeze_vit`` / ``freeze_aligner`` partition: by default only the language model is
adapted, leaving the vision tower and the aligner frozen. PEFT's own ``'all-linear'`` expansion has no
notion of towers, so it is replaced here by a regex anchored at the *unfrozen* towers' name prefixes.

The partition is passed in as a plain ``model_arch`` mapping -- ``language_model`` / ``vision_tower`` /
``aligner`` name-prefix lists plus an optional ``lm_head`` scalar -- which is serializable data threaded
from the caller's model loader. That keeps this model-agnostic: it reads the partition and walks the real
modules (``deep_getattr``) but hardcodes no family. The expansion itself runs worker-side, because under
Ray the driver only holds a PROXY model with no modules to inspect.

Ported from legacy swift's ``get_multimodal_target_regex`` / ``find_all_linears`` / ``find_layers``
(swift/utils/transformers_utils.py), with the ``model.model_meta`` / ``model.model_info`` reads replaced
by the explicit ``model_arch`` argument so nothing here depends on a swift model wrapper.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from torch import nn

from twinkle.utils import deep_getattr, get_logger

logger = get_logger()

__all__ = ['get_multimodal_target_regex', 'find_all_linears', 'find_layers']


def find_layers(model: nn.Module,
                cond: Callable[[str, nn.Module], bool],
                sub_module: Optional[str] = None,
                min_name_len: Optional[int] = None) -> List[str]:
    """The *leaf* module names under ``sub_module`` satisfying ``cond``, disambiguated against inner nodes.

    PEFT matches a list target by ``key.endswith('.' + name)``, so a bare leaf like ``qkv`` would also
    hit an unrelated ``qkv`` elsewhere. Walking the whole model first to collect the non-matching
    ("inner") node paths lets a leaf be lengthened with just enough parents to stay unique -- the same
    disambiguation legacy swift relies on. ``sub_module`` scopes the search to one tower while the names
    stay relative to it (the caller anchors them at the tower prefix).
    """
    sub_module_str = sub_module
    if sub_module is None:
        sub_module = model
    else:
        sub_module = deep_getattr(model, sub_module)
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r'\d+\.', '{}.', name)
        if not cond(name, module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in sub_module.named_modules():
        if sub_module_str:
            name = f'{sub_module_str}.{name}' if name else sub_module_str
        if cond(name, module):
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            i = 1
            for inner_node in inner_nodes:
                while module_name_list and inner_node.endswith(re.sub(
                        r'\d+\.', '{}.', module_name)) or min_name_len and i < min_name_len:
                    module_name = f'{module_name_list.pop()}.{module_name}'
                    i += 1
            target_module_names.add(module_name)
    return list(target_module_names)


def find_all_linears(model: nn.Module,
                     lm_head: Optional[str] = None,
                     extra_layers: Optional[List[type]] = None,
                     sub_module: Optional[str] = None) -> List[str]:
    """Every LoRA-able linear leaf under ``model`` (or ``sub_module``), by class-name heuristic.

    Matching on ``'linear' in class_name`` (rather than ``isinstance(nn.Linear)``) is deliberate: it also
    catches the parallel / fused linears a family defines in remote code. The output head and the
    classification / reward heads are excluded by name -- adapting them is both useless (they are the
    model's own read-out) and, for a freshly-initialized head, harmful. ``lm_head`` is the loader's
    declared head name, so a family that does not call it ``lm_head`` is still excluded.
    """
    if lm_head:
        lm_head_name = lm_head[lm_head.rfind('.') + 1:]
    else:
        lm_head_name = 'lm_head'
    # 'score' / 'classifier': sequence-classification head; 'v_head': reward-model head. The lora_* /
    # base_layer entries keep an already-adapted model from being targeted through its own wrappers.
    ignore_layers = [lm_head_name, 'score', 'v_head', 'classifier', 'lora_A', 'lora_B', 'base_layer']
    ignore_linear_cls = [
        'glulinear',  # phi4-mm
        'gemma4clippablelinear',  # gemma4
    ]

    def _cond(name: str, module: nn.Module) -> bool:
        module_name = module.__class__.__name__.lower()
        if (extra_layers and isinstance(module, tuple(extra_layers)) or
            ('linear' in module_name and all(linear_cls not in module_name
                                             for linear_cls in ignore_linear_cls))) and all(layer not in name
                                                                                            for layer in ignore_layers):
            return True
        return False

    return find_layers(model, _cond, sub_module=sub_module)


def get_multimodal_target_regex(model: nn.Module,
                                model_arch: Dict[str, Any],
                                *,
                                freeze_llm: bool = False,
                                freeze_vit: bool = True,
                                freeze_aligner: bool = True,
                                include_embedding: bool = False) -> str:
    """A PEFT target regex covering exactly the linear layers of the *unfrozen* towers.

    Each unfrozen tower contributes an alternative anchored at its name prefix (``re.escape(module)``
    followed by ``(?=\\.)``) and ending in that tower's linear leaf names. The aligner nests under the
    vision tower (``model.visual.merger`` under ``model.visual``), so when the vision tower is targeted
    the aligner prefixes are added as a negative lookahead -- the regex form of legacy's
    ``get_param_startswith(vision_tower, rejected=aligner)``, which keeps aligner params out of the vit
    group and lets the aligner's own alternative claim them when it is unfrozen.

    Raises when the flags freeze every tower (nothing left to adapt) or when the declared towers resolve
    to no linear at all (a ``model_arch`` that does not match the real model) -- both are louder than
    PEFT's downstream "no modules to save".
    """
    modules: List[str] = []
    if not freeze_llm:
        modules += list(model_arch.get('language_model') or [])
    if not freeze_vit:
        modules += list(model_arch.get('vision_tower') or [])
    if not freeze_aligner:
        modules += list(model_arch.get('aligner') or [])
    if not modules:
        raise ValueError('freeze_llm / freeze_vit / freeze_aligner are all True, so a multimodal '
                         "'all-linear' LoRA has no tower left to adapt. Unfreeze at least one of them.")

    aligner = list(model_arch.get('aligner') or [])
    lm_head = model_arch.get('lm_head')
    extra_layers: List[type] = [nn.Embedding] if include_embedding else []

    res = []
    for module in modules:
        rejected_modules = []
        if not freeze_vit or not freeze_llm:
            for aligner_prefix in aligner:
                if aligner_prefix.startswith(f'{module}.'):
                    rejected_modules.append(aligner_prefix)

        sub_module = deep_getattr(model, module)
        if sub_module is None:
            logger.warning(f'multimodal target expansion: module {module!r} resolved to None; skipped.')
            continue
        if isinstance(sub_module, nn.Linear) and module.endswith('lm_head'):
            # A tower entry that IS the output head (some loaders list lm_head under language_model):
            # never adapt the read-out head.
            target_modules: List[str] = []
        else:
            target_modules = find_all_linears(sub_module, lm_head, extra_layers)
        target_modules = [tm for tm in target_modules if tm]
        if not target_modules:
            continue
        target_pattern = rf'.*\.({"|".join(target_modules)})'
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''
        res.append(rf'{rejected_pattern}{re.escape(module)}(?=\.){target_pattern}')

    if not res:
        raise ValueError(
            f'multimodal target expansion found no linear layer under the unfrozen towers {modules}; '
            f'the threaded model_arch does not match this model. Check the loader\'s ModelArch.')
    return rf'^({"|".join(res)})$'
