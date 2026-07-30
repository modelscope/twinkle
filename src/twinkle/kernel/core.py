# Copyright (c) ModelScope Contributors. All rights reserved.
"""Mapping-driven kernel replacement.

``kernelize(model, mapping)`` installs the mapping entry by entry onto the
model: keys are replacement targets (HF class objects or dotted-path
strings), values are direct impls / ``HubRef`` / ``KernelChoice`` (picks an
impl by backend priority, see ``twinkle.kernel.registry``). How each entry
is installed is decided by the installer:
``KernelChoice.installer`` → ``OpDefinition.installer`` → ``default_installer``.

Public API: ``kernelize``, ``hub`` (re-exported from ``twinkle.kernel``).
"""
from __future__ import annotations

import importlib
import logging
import torch.nn as nn
from dataclasses import dataclass
from typing import Any

from twinkle import get_logger
from .registry import KernelChoice, get_op, resolve_impl

logger = get_logger()


@dataclass(frozen=True)
class HubRef:
    """Lightweight reference to a HuggingFace Hub kernel layer.

    Resolved lazily by ``kernelize`` via the optional ``kernels`` package.
    """
    repo_id: str
    layer_name: str
    revision: str | None = None
    version: int | None = None
    backend: str | None = None


def hub(
    ref: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> HubRef:
    """Build a ``HubRef`` for use as a ``kernelize`` mapping value.

    ``ref`` is ``'<repo_id>:<LayerName>'`` (e.g. ``'org/repo:SiluAndMul'``).
    Exactly one of ``revision`` or ``version`` must be supplied.
    """
    if (revision is None) == (version is None):
        raise ValueError('Exactly one of `revision` or `version` must be specified.')
    if ':' not in ref:
        raise ValueError(f"Hub ref must be 'repo_id:LayerName', got: {ref!r}")
    repo_id, layer_name = ref.rsplit(':', 1)
    return HubRef(repo_id, layer_name, revision, version, backend)


def _replace_class(model: nn.Module, target_cls: type, impl_cls: type) -> None:
    """Rewrite ``__class__`` of every module whose exact type is ``target_cls``.

    Uses ``type(m) is target_cls`` (not ``isinstance``) so user-defined
    subclasses of ``target_cls`` are deliberately left alone.
    """
    for m in model.modules():
        if type(m) is target_cls:
            m.__class__ = impl_cls


def _resolve_dotted(dotted_path: str) -> tuple[Any, str]:
    """Resolve ``pkg.mod[.Class].attr`` to ``(owner, final_attr)``.

    The split is found by walking the prefix from the longest importable
    module backwards until ``importlib.import_module`` succeeds; the remaining
    attributes (except the final one) are walked with ``getattr``.
    """
    parts = dotted_path.split('.')
    if len(parts) < 2:
        raise ValueError(f"Expected at least 'pkg.attr', got: {dotted_path!r}")

    # Find the longest prefix that imports as a module.
    last_err: ImportError | None = None
    module = None
    module_depth = 0
    for i in range(len(parts) - 1, 0, -1):
        candidate = '.'.join(parts[:i])
        try:
            module = importlib.import_module(candidate)
            module_depth = i
            break
        except ImportError as e:
            last_err = e
            continue
    if module is None:
        raise ImportError(f'Could not import any prefix of {dotted_path!r}') from last_err

    obj = module
    for attr in parts[module_depth:-1]:
        obj = getattr(obj, attr)
    return obj, parts[-1]


def _load_hub_ref(ref: HubRef):
    """Lazy-load a Hub kernel layer via the optional ``kernels`` package."""
    try:
        from kernels import get_kernel
    except ImportError as e:
        raise ImportError('Loading a Hub kernel requires the `kernels` package. '
                          'Install it with `pip install kernels`.') from e

    kernel = get_kernel(
        ref.repo_id,
        revision=ref.revision,
        version=ref.version,
        backend=ref.backend,
    )
    layers = getattr(kernel, 'layers', None)
    if layers is None:
        raise ValueError(f'Hub repo {ref.repo_id!r} does not define any layers.')
    impl = getattr(layers, ref.layer_name, None)
    if impl is None:
        raise ValueError(f'Layer {ref.layer_name!r} not found in {ref.repo_id!r}.')
    return impl


# ── installer ─────────────────────────────────────────────────────────────


def resolve_direct_value(replacement: Any) -> Any:
    """Non-KernelChoice mapping values: HubRef -> lazy download; anything else passes through unchanged."""
    if isinstance(replacement, HubRef):
        return _load_hub_ref(replacement)
    return replacement


def _install_dotted(model: nn.Module, target: str, impl, *, warn: bool = False) -> bool:
    """Dispatch a dotted-path target. Returns True = installed, False = family missing, skipped.

    - resolves to an ``nn.Module`` subclass -> ``_replace_class(model, cls, impl)``
      (exactly equivalent to a class-object key, exact type match)
    - otherwise -> ``setattr`` (module function / class method)
    - unresolvable ``transformers.*`` family path (missing module/attr) ->
      skip, return False (a missing family is normal). ``warn=False``
      (default config path) -> DEBUG; ``warn=True`` (explicit mapping) ->
      WARNING with a typo hint — explicit entries are the user's
      responsibility and a silent skip would hide spelling mistakes
    - unresolvable non-family string (e.g. a logical target mistakenly
      routed to the default installer) -> explicit error
    """
    try:
        owner, final_attr = _resolve_dotted(target)
        resolved = getattr(owner, final_attr)
    except (ImportError, AttributeError, ValueError) as e:
        if target.startswith('transformers.'):
            level = logging.WARNING if warn else logging.DEBUG
            hint = ' (explicit mapping: check for typos)' if warn else ''
            logger.log(level, '[kernelize] target %r unresolvable (%r); family not installed, skipping%s', target, e,
                       hint)
            return False
        raise ValueError(f'Cannot resolve mapping target {target!r} with the default installer '
                         f'(logical targets require a custom installer): {e!r}') from e
    if isinstance(resolved, type) and issubclass(resolved, nn.Module):
        _replace_class(model, resolved, impl)
    else:
        setattr(owner, final_attr, impl)
    return True


def default_installer(model: nn.Module, target: Any, impl, *, warn: bool = False) -> bool:
    """Generic installer: class objects -> ``_replace_class``; dotted-path strings -> ``_install_dotted``.

    ``warn`` is forwarded to ``_install_dotted`` for the family-skip log
    level; custom installers keep the ``(model, target, impl)`` three-arg
    signature, with kernelize dispatching between the two call shapes.
    """
    if isinstance(target, type) and issubclass(target, nn.Module):
        _replace_class(model, target, impl)
        return True
    if isinstance(target, str):
        return _install_dotted(model, target, impl, warn=warn)
    raise TypeError(f'Unsupported mapping target: {target!r}')


# ── kernelize ─────────────────────────────────────────────────────────────


def _target_name(target: Any) -> str:
    if isinstance(target, type):
        return f'{target.__module__}.{target.__qualname__}'
    return str(target)


def _installer_name(installer) -> str:
    if installer is default_installer:
        return 'default'
    return getattr(installer, '__name__', repr(installer))


def _log_all_unavailable(target: Any, choice: KernelChoice, *, warn: bool) -> None:
    level = logging.WARNING if warn else logging.DEBUG
    logger.log(
        level, "[kernelize] target %s: no available backend for op '%s' (tried: %s); "
        'keeping the original implementation', _target_name(target), choice.op, ', '.join(choice.backends))


def kernelize(model: nn.Module, mapping: dict | None = None) -> nn.Module:
    """Apply ``mapping`` to ``model`` and return it (modified in place).

    Keys (targets):
      - ``type[nn.Module]``: replace ``m.__class__`` for every module of the
        exact type (no subclass walking).
      - ``str`` dotted path: resolved by the default installer — an
        ``nn.Module`` subclass resolves to class replacement, anything else
        (module function / class method) to ``setattr``. Unresolvable
        ``transformers.*`` family paths are skipped: DEBUG on the default
        path, WARNING (with typo hint) for explicit mappings.

    Values:
      - ``KernelChoice``: pick the first available backend impl for the op
        (see ``twinkle.kernel.registry``); installer priority is
        ``KernelChoice.installer`` → ``OpDefinition.installer`` → default.
      - ``HubRef``: lazy-resolved via the optional ``kernels`` package.
      - anything else: used directly as the impl (default installer).

    ``mapping=None`` applies the built-in ``DEFAULT_KERNEL_CONFIG``; on this
    default path all fallback/skip logs are DEBUG. Passing any explicit
    mapping (even a copy of the default) raises them to WARNING. The mapping
    fully *replaces* the default config — merge with
    ``{**DEFAULT_KERNEL_CONFIG, ...}`` to customize on top of it.
    """
    default_mapping = mapping is None
    if default_mapping:
        from .config import DEFAULT_KERNEL_CONFIG  # in-function import, avoids the core<->config cycle
        mapping = DEFAULT_KERNEL_CONFIG

    if not mapping:
        return model

    warn = not default_mapping
    for target, replacement in mapping.items():
        if isinstance(replacement, KernelChoice):
            op = get_op(replacement.op)  # unregistered -> ValueError
            impl, backend = resolve_impl(op, replacement.backends, warn=warn, target=target)
            if impl is None:
                _log_all_unavailable(target, replacement, warn=warn)
                continue  # keep the original implementation
            installer = replacement.installer or op.installer or default_installer
        else:
            impl = resolve_direct_value(replacement)
            backend = None
            installer = default_installer

        # custom installers keep the three-arg signature; warn is only forwarded to default_installer
        if installer is default_installer:
            installed = installer(model, target, impl, warn=warn)
        else:
            installed = installer(model, target,
                                  impl)  # failures propagate, never swallowed (half-installed state stays visible)
        if installed is False:
            continue
        if isinstance(replacement, KernelChoice):
            logger.info('[kernelize] target=%s op=%s backend=%s installer=%s', _target_name(target), op.name, backend,
                        _installer_name(installer))
        else:
            impl_repr = getattr(impl, '__qualname__', repr(impl))
            logger.info('[kernelize] target=%s impl=%s installer=%s', _target_name(target), impl_repr,
                        _installer_name(installer))
    return model
