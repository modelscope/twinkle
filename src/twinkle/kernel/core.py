# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal mapping-driven kernel replacement.

Public API: ``kernelize``, ``hub`` (re-exported from ``twinkle.kernel``).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


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


def _infer_device(model: nn.Module) -> str:
    """Infer the device type from the first parameter, then first buffer, else cpu."""
    for p in model.parameters():
        return p.device.type
    for b in model.buffers():
        return b.device.type
    return 'cpu'


def _resolve_value(value: Any, device: str) -> Any | None:
    """Resolve a mapping value against the inferred device.

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
    """``setattr`` ``impl`` onto the module identified by the dotted path's prefix."""
    module_path, _, attr = dotted_path.rpartition('.')
    if not module_path or not attr:
        raise ValueError(f"Expected 'pkg.module.attr', got: {dotted_path!r}")
    module = importlib.import_module(module_path)
    setattr(module, attr, impl)


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