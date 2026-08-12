# Copyright (c) ModelScope Contributors. All rights reserved.
"""Operator implementation registry: lazy loading and availability of each
backend's impl per op, plus KernelChoice resolution and fallback.

Only "implementations" are registered here, never "targets" — targets live in
config.py's default mapping and in user mappings.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable

from twinkle import get_logger
from twinkle.utils.device_mesh import Platform
from twinkle.utils.import_utils import exists

logger = get_logger()

__all__ = [
    'KernelImpl',
    'OpDefinition',
    'KernelChoice',
    'Installer',
    'register_op',
    'get_op',
    'resolve_impl',
    'lazy_import',
    'is_npu_available',
    'is_liger_available',
    'exists',
]

# ── Data structures ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class KernelImpl:
    """One backend's implementation of one op.

    ``load``: lazily loads and returns the final class or function; impls
    must not import optional deps (``torch_npu`` / ``liger_kernel`` ...) at
    registration time. Receives the mapping target (dotted-path string or
    class object) so a backend can specialize per target family (e.g. liger's
    RMSNorm dispatches gemma/qwen3_5 variants); factories that don't care
    simply ignore it (see ``lazy_import``).
    ``available``: whether the current platform/deps/hardware permit this
    impl; ``(True, None)`` = usable, ``(False, reason)`` = not usable, fall
    through to the next backend.
    """
    load: Callable[[Any], Any]
    available: Callable[[], tuple[bool, str | None]]


Installer = Callable[[nn.Module | None, Any, Any], None]  # (model, target, impl) -> None


@dataclass(frozen=True)
class OpDefinition:
    """All backend implementations of one op, plus its default installer."""
    name: str  # 'swiglu' | 'rms_norm' | 'sdpa_attention' | ...
    implementations: dict[str, KernelImpl]  # backend name -> impl
    installer: Installer | None = None  # None = use the generic installer


@dataclass(frozen=True)
class KernelChoice:
    """Selection descriptor in a mapping: which op to use and the backend priority."""
    op: str  # references a registered OpDefinition
    backends: tuple[str, ...]  # priority-ordered, at least one element
    installer: Installer | None = None  # advanced override; usually None


# installer priority: KernelChoice.installer -> OpDefinition.installer -> default_installer

# ── Registry ──────────────────────────────────────────────────────────────

_OPS: dict[str, OpDefinition] = {}


def register_op(
    name: str,
    *,
    implementations: dict[str, KernelImpl],
    installer: Installer | None = None,
) -> None:
    """Register an op. Duplicate name / empty implementations -> ValueError."""
    if name in _OPS:
        raise ValueError(f'op {name!r} is already registered')
    if not implementations:
        raise ValueError(f'op {name!r} has no implementations')
    _OPS[name] = OpDefinition(name=name, implementations=dict(implementations), installer=installer)


def get_op(name: str) -> OpDefinition:
    """Fetch an op definition; unregistered -> ValueError (includes the op name)."""
    try:
        return _OPS[name]
    except KeyError:
        raise ValueError(f'op {name!r} is not registered') from None


def resolve_impl(
    op: OpDefinition,
    backends: tuple[str, ...],
    *,
    warn: bool,
    target: Any = None,
) -> tuple[Any, str | None]:
    """Pick the first available impl in ``backends`` order.

    In turn: backend not registered -> log and skip; available() False ->
    log (with reason) and skip; load() raises -> log (with the exception)
    and skip; first success -> (impl, backend_name). All failed ->
    (None, None): no installer call, original implementation kept.

    warn=True (explicit mapping) -> fallback/failure logs at WARNING;
    warn=False (default config path) -> all DEBUG.
    ``target`` is passed verbatim to KernelImpl.load() for family-specialized
    dispatch.
    """
    level = logging.WARNING if warn else logging.DEBUG
    for backend in backends:
        impl_entry = op.implementations.get(backend)
        if impl_entry is None:
            logger.log(level, "[kernelize] op '%s': backend '%s' not registered, skipping", op.name, backend)
            continue
        ok, reason = impl_entry.available()
        if not ok:
            logger.log(level, "[kernelize] op '%s': backend '%s' unavailable (%s), skipping", op.name, backend, reason)
            continue
        try:
            return impl_entry.load(target), backend
        except Exception as e:
            logger.log(level, "[kernelize] op '%s': backend '%s' failed to load (%r), skipping", op.name, backend, e)
    return None, None


# ── Lazy references and availability helpers (shared by ops/*/__init__.py) ─


def lazy_import(spec: str) -> Callable[[Any], Any]:
    """'pkg.mod:attr' -> lazy-load factory; imports the module and getattr
    only when called.

    Missing module / attribute -> raises ImportError/AttributeError, caught
    by resolve_impl's load() exception branch (falls through).
    The factory accepts (and ignores) the mapping target argument, matching
    the KernelImpl.load signature.
    """
    module_path, _, attr = spec.partition(':')

    def _load(_target: Any = None) -> Any:
        return getattr(importlib.import_module(module_path), attr)

    return _load


def is_npu_available() -> tuple[bool, str | None]:
    """Platform is npu and torch_npu is importable."""
    if Platform.device_prefix() != 'npu':
        return False, f"platform is '{Platform.device_prefix()}', not 'npu'"
    if importlib.util.find_spec('torch_npu') is None:
        return False, 'torch_npu not installed'
    return True, None


def is_liger_available() -> tuple[bool, str | None]:
    """liger_kernel is importable (find_spec only, no real import)."""
    if importlib.util.find_spec('liger_kernel') is None:
        return False, 'liger_kernel not installed'
    return True, None
