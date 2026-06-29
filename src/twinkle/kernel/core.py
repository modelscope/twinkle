# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal mapping-driven kernel replacement.

Public API: ``kernelize``, ``hub`` (re-exported from ``twinkle.kernel``).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from twinkle.utils.device_mesh import Platform


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
    trust_remote_code: bool = False


def hub(
    ref: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    trust_remote_code: bool = False,
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
    return HubRef(repo_id, layer_name, revision, version, backend, trust_remote_code)



def _resolve_value(value: Any, device: str) -> Any | None:
    """Resolve a mapping value against the selected device.

    - ``dict``: device-conditional; recurse into ``value[device]`` or return None.
    - anything else (including ``HubRef``): pass through.
    """
    if isinstance(value, dict):
        if device not in value:
            return None
        return _resolve_value(value[device], device)
    return value


def _replace_class(model: nn.Module, target_cls: type, impl_cls: type) -> None:
    """Rewrite ``__class__`` of every module whose exact type is ``target_cls``.

    Uses ``type(m) is target_cls`` (not ``isinstance``) so user-defined
    subclasses of ``target_cls`` are deliberately left alone.
    """
    for m in model.modules():
        if type(m) is target_cls:
            m.__class__ = impl_cls


def _replace_attr(dotted_path: str, impl) -> None:
    """``setattr`` ``impl`` onto the attribute identified by the dotted path.

    Supports two forms:
      - ``pkg.mod.attr``                 (set module attribute)
      - ``pkg.mod.ClassName.attr``       (set class attribute / method)

    The split is found by walking the prefix from the longest importable
    module backwards until ``importlib.import_module`` succeeds.
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

    # Walk remaining attributes; the last one is the target.
    obj = module
    for attr in parts[module_depth:-1]:
        obj = getattr(obj, attr)
    setattr(obj, parts[-1], impl)


def _load_hub_ref(ref: HubRef):
    """Lazy-load a Hub kernel layer via the optional ``kernels`` package."""
    try:
        from kernels import get_kernel
    except ImportError as e:
        raise ImportError(
            'Loading a Hub kernel requires the `kernels` package. '
            'Install it with `pip install kernels`.'
        ) from e

    kernel = get_kernel(
        ref.repo_id,
        revision=ref.revision,
        version=ref.version,
        backend=ref.backend,
        trust_remote_code=ref.trust_remote_code,
    )
    layers = getattr(kernel, 'layers', None)
    if layers is None:
        raise ValueError(f'Hub repo {ref.repo_id!r} does not define any layers.')
    impl = getattr(layers, ref.layer_name, None)
    if impl is None:
        raise ValueError(f'Layer {ref.layer_name!r} not found in {ref.repo_id!r}.')
    return impl


def kernelize(model: nn.Module, mapping: dict) -> nn.Module:
    """Apply ``mapping`` to ``model`` and return it (modified in place).

    Keys:
      - ``type[nn.Module]``: replace ``m.__class__`` for every module of the
        exact type (no subclass walking).
      - ``str`` (dotted path ``pkg.mod.attr``): ``setattr`` the impl onto the
        identified module attribute.

    Values:
      - ``dict[str, V]``: device-conditional dispatch using the current
        Twinkle platform device prefix; non-matching devices skip.
      - ``HubRef``: lazy-resolved via the optional ``kernels`` package.
      - anything else: used directly as the impl.
    """
    if not mapping:
        return model

    device = Platform.device_prefix()
    for key, value in mapping.items():
        impl = _resolve_value(value, device)
        if impl is None:
            continue
        if isinstance(impl, HubRef):
            impl = _load_hub_ref(impl)
        if isinstance(key, type) and issubclass(key, nn.Module):
            _replace_class(model, key, impl)
        elif isinstance(key, str):
            _replace_attr(key, impl)
        else:
            raise TypeError(f'Unsupported mapping key: {key!r}')
    return model
