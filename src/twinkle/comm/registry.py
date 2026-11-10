# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend name -> :class:`Communicator` implementation, plus the ``create_communicator`` entry.

Backends are registered as ``'module:ClassName'`` strings and imported on first use, so this
module imports on any host regardless of which accelerator stack is installed.
"""
from __future__ import annotations
import importlib
from datetime import timedelta
from typing import Any

import torch

from .base import DEFAULT_TIMEOUT, Communicator, Rendezvous

_BACKENDS: dict[str, str | type[Communicator]] = {
    'gloo': 'twinkle.comm.backends.gloo:GlooCommunicator',
}


def _check_communicator_cls(name: str, target: Any) -> None:
    if not (isinstance(target, type) and issubclass(target, Communicator)):
        raise TypeError(f'communicator backend {name!r} must be a Communicator subclass, got {target!r}')


def register_backend(name: str, target: str | type[Communicator], *, override: bool = False) -> None:
    """Register ``target`` (a class or a lazy ``'module:ClassName'`` path) under ``name``."""
    if name in _BACKENDS and not override:
        raise ValueError(f'communicator backend {name!r} is already registered')
    if not isinstance(target, str):
        _check_communicator_cls(name, target)
    elif ':' not in target:
        raise ValueError(f"lazy backend path must look like 'module:ClassName', got {target!r}")
    _BACKENDS[name] = target


def unregister_backend(name: str) -> None:
    if _BACKENDS.pop(name, None) is None:
        raise ValueError(f'no communicator backend registered as {name!r}')


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def default_backend() -> str:
    """The backend the current platform communicates over (``Platform.communicator_backend``)."""
    from twinkle.utils.platforms import Platform
    return Platform.get_platform().communicator_backend()


def get_communicator_cls(backend: str | None = None) -> type[Communicator]:
    """Resolve ``backend`` (default: the current platform's) to its implementation class."""
    name = backend or default_backend()
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(f'no communicator backend registered as {name!r}; '
                         f'available: {available_backends()}') from None
    if isinstance(target, str):
        module_name, _, attr = target.partition(':')
        try:
            target = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as error:
            raise ImportError(f'communicator backend {name!r} ({module_name}:{attr}) could not be loaded; '
                              f'is its accelerator stack installed? ({error})') from error
        _check_communicator_cls(name, target)
        _BACKENDS[name] = target
    return target


def create_communicator(
    rendezvous: Rendezvous,
    rank: int,
    world_size: int,
    device: torch.device | str,
    backend: str | None = None,
    *,
    timeout: timedelta = DEFAULT_TIMEOUT,
    **options,
) -> Communicator:
    """Build a standalone communication group. Collective over all ``world_size`` ranks.

    Args:
        rendezvous: Address (and key namespace) rank 0 hosts the rendezvous store on; identical
            on every rank.
        rank: This process's rank inside the group (``0 .. world_size - 1``), not its global rank.
        world_size: Number of processes in the group; identical on every rank.
        device: Device the communicator binds to; tensors passed to it must live there.
            ``'cuda'`` / ``'npu'`` without an index resolve to the current device.
        backend: Backend name; ``None`` lets the current platform decide.
        timeout: Upper bound for building the group and for each operation.
        **options: Backend-specific settings, checked against the backend's ``supported_options``.
    """
    return get_communicator_cls(backend).create(rendezvous, rank, world_size, device, timeout=timeout, **options)
