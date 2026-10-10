# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-neutral application of an optimizer param-group *spec*.

dev's ``swift.dev.optimizer.build_param_groups_spec`` emits a serializable, ordered list of rules --
first match wins, most specific first -- and hands it to ``set_optimizer`` as ``param_groups_spec``.
This module is the shared kernel both twinkle backends use to turn that spec into their local
param-group form, so the two stay equivalent and twinkle hardcodes no model / tuner / tower naming:

- transformers: :func:`apply_param_groups_spec` splits the decay / no-decay groups built by
  ``TransformersModel._create_param_group`` into per-rule sub-groups, each keeping its ``param_names``
  (checkpointing relies on them).
- megatron: :func:`build_param_group_overrides` translates each rule into an mcore
  ``{ParamKey: ParamGroupOverride}`` entry, to be layered on top of ``get_standard_config_overrides``.

A rule is a plain dict::

    {
      'match': {                    # all present keys are AND-ed
          'tower': 'vision_tower',  # role, resolved to name prefixes by the backend (see below)
          'name': '*lora_B*',       # fnmatch glob, or a list of globs (OR), on the logical name
          'ndim': 1,                # param.ndim
          'module_type': 'Embedding',
      },
      'lr': 1e-5,                   # absolute lr; mutually exclusive with lr_mult
      'lr_mult': 2.0,               # or: multiple of the group's base lr
      'weight_decay': 0.0,          # optional; omitted/None inherits the group's decay class
    }

The ``tower`` role keeps the spec backend-neutral: each backend resolves it to the name prefixes its
own parameters use and passes them in as ``tower_prefixes`` (transformers from the threaded
``ModelArch``; megatron from the live model's ``visual._vision_tower`` / ``_aligner``). Matching
therefore runs worker-side against the real parameters -- the driver only ever ships data, which is
what lets the customization cross the Ray boundary (the driver's model is a PROXY with no parameters).

mcore *merges* every matching ``ParamKey`` per parameter rather than taking the first, so
:func:`build_param_group_overrides` makes each rule exclusive -- its predicate fires only when that
rule is the parameter's first match -- which is what reproduces first-match-wins on megatron.
"""
from __future__ import annotations

import fnmatch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from torch import nn

from twinkle.utils import deep_getattr, get_logger

logger = get_logger()

__all__ = [
    'apply_param_groups_spec', 'build_param_group_overrides', 'validate_param_groups_spec', 'resolve_tower_prefixes',
    'resolve_megatron_tower_prefixes'
]

#: The tower roles a rule may reference. Every backend resolves all three (to an empty prefix list
#: when the model has no such part) so a rule never fails on one backend and works on the other.
TOWER_ROLES = ('vision_tower', 'aligner', 'language_model')
_MATCH_KEYS = frozenset({'tower', 'name', 'ndim', 'module_type'})
_RULE_KEYS = frozenset({'match', 'lr', 'lr_mult', 'weight_decay'})


def _logical_name(name: str) -> str:
    """Strip the wrappers each backend adds, down to the path the prefixes / globs are written against.

    Megatron wraps every chunk in DDP + Float16Module (two leading ``module.``); a peft-wrapped
    transformers model carries ``base_model.model.``. Removing them yields the same logical module
    path a plain ``named_parameters()`` on the unwrapped model would give, which is what ``ModelArch``
    (transformers) and the ``visual.``-prefixed megatron tower names are expressed in.
    """
    name = name.removeprefix('module.').removeprefix('module.')
    return name.removeprefix('base_model.model.')


def _match_name(logical: str, name_spec: Any) -> bool:
    patterns = [name_spec] if isinstance(name_spec, str) else list(name_spec)
    return any(fnmatch.fnmatch(logical, pattern) for pattern in patterns)


def _match_tower(logical: str, role: str, tower_prefixes: Dict[str, List[str]]) -> bool:
    return any(logical.startswith(prefix) for prefix in tower_prefixes[role])


def _transformers_module_type_resolver(model) -> Callable[[Any, str, str], bool]:
    """Resolve ``module_type`` by walking to the parameter's parent module and isinstance-checking it.

    Mirrors legacy swift's LoRA+ ``get_module``: a lora parameter's grandparent is the wrapped module
    (drop the ``.lora_X.<adapter>`` tail), any other parameter's parent is one level up (drop the
    ``.weight`` / ``.bias`` tail). ``raw_name`` is used -- not the logical name -- so the walk root and
    the name stay consistent with the ``named_parameters()`` that produced it.
    """

    def resolve(param: Any, raw_name: str, module_type: str) -> bool:
        module_cls = getattr(nn, module_type, None)
        if not isinstance(module_cls, type):
            raise ValueError(
                f"param_groups module_type {module_type!r} is not a torch.nn class; the transformers "
                f"backend resolves module_type by isinstance against torch.nn.<module_type>.")
        drop = 2 if 'lora' in raw_name else 1
        parent_path = '.'.join(raw_name.split('.')[:-drop])
        parent = deep_getattr(model, parent_path) if parent_path else None
        return isinstance(parent, module_cls)

    return resolve


def _megatron_module_type_resolver() -> Callable[[Any, str, str], bool]:
    """Resolve ``module_type`` the megatron-native way: mcore tags params instead of walking modules.

    A ``with_name_predicate`` only receives ``(param, name)`` and megatron shards modules across
    pipeline ranks, so there is no single root to walk. mcore already marks the word-embedding /
    output-layer weights with ``is_embedding_or_output_parameter`` (the same attribute its decoupled-lr
    path uses), which is the megatron equivalent of an ``isinstance(nn.Embedding)`` parent check.
    """

    def resolve(param: Any, raw_name: str, module_type: str) -> bool:
        if module_type == 'Embedding':
            return bool(getattr(param, 'is_embedding_or_output_parameter', False))
        raise ValueError(
            f"param_groups module_type {module_type!r} cannot be resolved on the megatron backend; "
            f"only 'Embedding' is supported there (via is_embedding_or_output_parameter).")

    return resolve


def _rule_matches(rule: Dict[str, Any], *, logical: str, param: Any, raw_name: str,
                  tower_prefixes: Dict[str, List[str]],
                  resolve_module_type: Callable[[Any, str, str], bool]) -> bool:
    match = rule['match']
    if 'tower' in match and not _match_tower(logical, match['tower'], tower_prefixes):
        return False
    if 'name' in match and not _match_name(logical, match['name']):
        return False
    if 'ndim' in match and param.ndim != match['ndim']:
        return False
    if 'module_type' in match and not resolve_module_type(param, raw_name, match['module_type']):
        return False
    return True


def _first_match(spec: Sequence[Dict[str, Any]], *, logical: str, param: Any, raw_name: str,
                 tower_prefixes: Dict[str, List[str]],
                 resolve_module_type: Callable[[Any, str, str], bool]) -> Optional[int]:
    for index, rule in enumerate(spec):
        if _rule_matches(rule,
                         logical=logical,
                         param=param,
                         raw_name=raw_name,
                         tower_prefixes=tower_prefixes,
                         resolve_module_type=resolve_module_type):
            return index
    return None


def _rule_lr(rule: Dict[str, Any], base_lr: Optional[float]) -> Optional[float]:
    if 'lr' in rule:
        return float(rule['lr'])
    if 'lr_mult' in rule:
        return float(base_lr) * float(rule['lr_mult'])
    return base_lr


def validate_param_groups_spec(spec: Any) -> None:
    """Reject a malformed spec loudly, on both backends, before any parameter is inspected."""
    if not isinstance(spec, list):
        raise TypeError(f'param_groups_spec must be a list of rule dicts, got {type(spec).__name__}')
    for index, rule in enumerate(spec):
        if not isinstance(rule, dict):
            raise ValueError(f'param_groups rule {index} must be a dict, got {type(rule).__name__}')
        unknown = set(rule) - _RULE_KEYS
        if unknown:
            raise ValueError(f'param_groups rule {index} has unknown key(s) {sorted(unknown)}')
        match = rule.get('match')
        if not isinstance(match, dict) or not match:
            raise ValueError(f'param_groups rule {index} must have a non-empty "match" dict')
        unknown = set(match) - _MATCH_KEYS
        if unknown:
            raise ValueError(f'param_groups rule {index} match has unknown key(s) {sorted(unknown)}')
        if 'lr' in rule and 'lr_mult' in rule:
            raise ValueError(f'param_groups rule {index} sets both "lr" and "lr_mult"; they are exclusive')
        if 'lr' not in rule and 'lr_mult' not in rule and rule.get('weight_decay') is None:
            raise ValueError(f'param_groups rule {index} changes neither lr nor weight_decay; nothing to apply')


def _check_tower_roles(spec: Sequence[Dict[str, Any]], tower_prefixes: Dict[str, List[str]]) -> None:
    for index, rule in enumerate(spec):
        role = rule['match'].get('tower')
        if role is not None and role not in tower_prefixes:
            raise ValueError(
                f'param_groups rule {index} references tower role {role!r}, which this backend does not '
                f'resolve (known roles: {sorted(tower_prefixes)}).')


def _rule_is_noop(rule: Dict[str, Any], base_lr: Optional[float]) -> bool:
    """True when a rule re-assigns the base lr and inherits weight decay, so it changes nothing.

    Such a rule is emitted only to *claim* parameters for ordering -- e.g. the aligner rule that keeps
    a nested aligner's params out of the broader vision_tower rule under first-match-wins -- not to move
    any learning rate, so matching zero of them is never worth surfacing.
    """
    if rule.get('weight_decay') is not None:
        return False
    if 'lr_mult' in rule:
        return float(rule['lr_mult']) == 1.0
    if 'lr' in rule:
        return base_lr is not None and float(rule['lr']) == float(base_lr)
    return True


def _warn_unmatched(spec: Sequence[Dict[str, Any]], matched: Sequence[int],
                    tower_prefixes: Dict[str, List[str]], base_lr: Optional[float] = None) -> None:
    """Surface a per-tower learning rate that caught nothing.

    Only ``tower`` rules are checked: each maps one-to-one onto a user knob (``vit_lr`` / ``aligner_lr``),
    so an empty match means that knob had no effect -- either the tower is frozen (``freeze_vit`` left it
    untrained) or the model has no such part. Finer-grained ``name`` / ``ndim`` / ``module_type`` rules
    stay silent when empty: LoRA+'s embedding and ``ndim == 1`` sub-groups are legitimately empty in an
    ordinary LoRA run (no embedding rows or length-1 params are trained), so warning on them would be
    noise -- this matches legacy swift, which simply skips an empty group. A no-op rule (one that only
    re-assigns the base lr) is silent for the same reason.
    """
    for index, rule in enumerate(spec):
        if matched[index] > 0:
            continue
        role = rule['match'].get('tower')
        if role is None or _rule_is_noop(rule, base_lr):
            continue
        if tower_prefixes.get(role):
            logger.warning(
                f'param_groups rule {index} sets a {role} learning rate but matched 0 trainable '
                f'parameters, so it has no effect. That tower is frozen -- check freeze_vit / '
                f'freeze_aligner / freeze_llm and target_modules.')
        else:
            logger.warning(f'param_groups rule {index} sets a {role} learning rate but this model has no '
                           f'{role} part, so it has no effect.')


def resolve_tower_prefixes(model_arch: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Resolve the tower roles from a serializable ModelArch mapping (transformers backend).

    ``model_arch`` is the plain ``{'vision_tower': [...], 'aligner': [...], 'language_model': [...]}``
    dict threaded from dev's loader -- HF module-name prefixes, which is exactly the space a transformers
    ``named_parameters()`` (after :func:`_logical_name`) lives in. All three roles are always present
    (empty when the model has no such part) so a spec resolves identically on both backends.
    """
    model_arch = model_arch or {}
    return {role: list(model_arch.get(role) or []) for role in TOWER_ROLES}


def resolve_megatron_tower_prefixes(model_chunks: Sequence[Any]) -> Dict[str, List[str]]:
    """Resolve the tower roles from a live megatron model (mcore module-name space).

    Megatron parameters are named in mcore space, not HF space, so the transformers ``ModelArch``
    prefixes do not transfer. mcore-bridge declares the partition on the model instead: a multimodal
    chunk is a ``MultimodalGPTModel`` whose LLM sits under ``language_model`` and whose
    ``model_meta.visual_cls`` carries ``_vision_tower`` / ``_aligner`` (relative to ``visual``) as
    *class* attributes -- the same source legacy swift megatron reads. Being class attributes, they
    resolve on every pipeline rank even where the ``visual`` instance itself is ``None`` (only the
    first stage builds it). A plain ``GPTModel`` has no ``visual_cls``, so every prefix stays empty and
    no tower rule matches -- the correct "treat the whole model uniformly" behavior for a plain LLM.
    """
    prefixes: Dict[str, List[str]] = {role: [] for role in TOWER_ROLES}
    for chunk in model_chunks:
        core = chunk
        while hasattr(core, 'module'):  # unwrap DDP / Float16Module down to the mcore model
            core = core.module
        visual_cls = getattr(getattr(core, 'model_meta', None), 'visual_cls', None)
        if visual_cls is None:
            continue
        prefixes['language_model'] = ['language_model']
        prefixes['vision_tower'] = [f'visual.{name}' for name in (getattr(visual_cls, '_vision_tower', None) or [])]
        prefixes['aligner'] = [f'visual.{name}' for name in (getattr(visual_cls, '_aligner', None) or [])]
        break
    return prefixes


def apply_param_groups_spec(groups: List[dict],
                            spec: Sequence[Dict[str, Any]],
                            *,
                            model,
                            tower_prefixes: Dict[str, List[str]]) -> List[dict]:
    """Split transformers param groups by the spec, preserving each group's decay class.

    Runs after ``_create_param_group`` (so the decay / no-decay split and ``param_names`` exist) and
    before GaLore / Muon reshaping. Every group is partitioned by first-match-wins into a base bucket
    (no rule matched -> untouched lr / weight_decay) plus one bucket per matched rule, so the tower x
    decay cross-product falls out naturally and matches legacy ``MultimodalOptimizerCallback`` /
    LoRA+ grouping.
    """
    validate_param_groups_spec(spec)
    _check_tower_roles(spec, tower_prefixes)
    resolve_module_type = _transformers_module_type_resolver(model)
    # Every group _create_param_group builds carries the same optimizer lr, so the first is the base
    # the no-op check compares a rule's absolute lr against.
    base_lr = groups[0].get('lr') if groups else None
    matched = [0] * len(spec)
    new_groups: List[dict] = []
    for group in groups:
        if 'param_names' not in group:
            raise ValueError(
                'param_groups_spec needs param groups carrying "param_names" (as built by '
                'TransformersModel._create_param_group); a group without names cannot be split by '
                'name / tower / module_type.')
        base_lr = group.get('lr')
        base_wd = group.get('weight_decay', 0.0)
        buckets: Dict[int, Tuple[List[Any], List[str]]] = {}
        for raw_name, param in zip(group['param_names'], group['params']):
            rule_index = _first_match(spec,
                                      logical=_logical_name(raw_name),
                                      param=param,
                                      raw_name=raw_name,
                                      tower_prefixes=tower_prefixes,
                                      resolve_module_type=resolve_module_type)
            key = -1 if rule_index is None else rule_index
            if rule_index is not None:
                matched[rule_index] += 1
            params_bucket, names_bucket = buckets.setdefault(key, ([], []))
            params_bucket.append(param)
            names_bucket.append(raw_name)
        # Base bucket first (the "everything else" group), then rule buckets in spec order.
        for key in (-1, *range(len(spec))):
            if key not in buckets:
                continue
            params_bucket, names_bucket = buckets[key]
            if key == -1:
                new_groups.append({
                    'params': params_bucket,
                    'param_names': names_bucket,
                    'weight_decay': base_wd,
                    'lr': base_lr,
                })
            else:
                rule = spec[key]
                weight_decay = rule.get('weight_decay')
                new_groups.append({
                    'params': params_bucket,
                    'param_names': names_bucket,
                    'weight_decay': base_wd if weight_decay is None else weight_decay,
                    'lr': _rule_lr(rule, base_lr),
                })
    _warn_unmatched(spec, matched, tower_prefixes, base_lr)
    return new_groups


def build_param_group_overrides(spec: Sequence[Dict[str, Any]],
                                *,
                                model_chunks: Sequence[Any],
                                tower_prefixes: Dict[str, List[str]],
                                base_lr: float,
                                base_min_lr: float = 0.0) -> Dict[Any, Dict[str, Any]]:
    """Translate the spec into mcore ``{ParamKey: ParamGroupOverride}`` for the megatron backend.

    The caller layers the result on top of ``get_standard_config_overrides(config)`` so per-tower /
    LoRA+ learning rates do not drop mcore's default bias / length-1 weight-decay skip: the two sets
    of keys merge per parameter (an mcore ``ParamKey`` is not first-match-wins), which is also why each
    rule's predicate fires only when that rule is the parameter's *first* match -- exclusivity is what
    reproduces the spec's ordering under merge semantics.

    ``min_lr`` is scaled by the same ratio as ``max_lr`` so the whole schedule moves together, matching
    legacy swift megatron (``_lr_mult = vit_lr / lr`` applied to the schedule, not just the peak).
    """
    from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate

    validate_param_groups_spec(spec)
    _check_tower_roles(spec, tower_prefixes)
    resolve_module_type = _megatron_module_type_resolver()

    # The predicate is applied lazily by mcore, so count matches eagerly here to warn on dead rules
    # exactly like the transformers applier does (only this rank's trainable params are visible).
    matched = [0] * len(spec)
    for chunk in model_chunks:
        for raw_name, param in chunk.named_parameters():
            if not param.requires_grad:
                continue
            rule_index = _first_match(spec,
                                      logical=_logical_name(raw_name),
                                      param=param,
                                      raw_name=raw_name,
                                      tower_prefixes=tower_prefixes,
                                      resolve_module_type=resolve_module_type)
            if rule_index is not None:
                matched[rule_index] += 1
    _warn_unmatched(spec, matched, tower_prefixes, base_lr)

    overrides: Dict[Any, Dict[str, Any]] = {}
    for index, rule in enumerate(spec):

        def make_predicate(rule_index: int) -> Callable[[Any, str], bool]:

            def predicate(param: Any, name: str) -> bool:
                first = _first_match(spec,
                                     logical=_logical_name(name),
                                     param=param,
                                     raw_name=name,
                                     tower_prefixes=tower_prefixes,
                                     resolve_module_type=resolve_module_type)
                return first == rule_index

            return predicate

        group_lr = _rule_lr(rule, base_lr)
        override: Dict[str, Any] = {'max_lr': group_lr}
        if base_lr:
            override['min_lr'] = base_min_lr * (group_lr / base_lr)
        weight_decay = rule.get('weight_decay')
        if weight_decay is not None:
            # Absolute wd: pin start/end and neutralise wd_mult so a merged standard skip (wd_mult=0)
            # cannot zero out an explicitly requested decay.
            override['start_wd'] = override['end_wd'] = weight_decay
            override['wd_mult'] = 1.0
        key = ParamKey(with_name_predicate=ParamWithNamePredicate(
            name=f'param_groups_rule_{index}', fn=make_predicate(index)))
        overrides[key] = override
    return overrides
