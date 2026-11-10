# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend name -> :class:`Communicator` implementation, plus the ``create_communicator`` entry.

Backends are registered as ``'module:ClassName'`` strings and imported on first use, so this
module imports on any host regardless of which accelerator stack is installed.
"""
from __future__ import annotations
import importlib

import torch

from .base import Communicator, Rendezvous

_BACKENDS: dict[str, str | type[Communicator]] = {
    'gloo': 'twinkle.comm.backends.gloo:GlooCommunicator',
}


def register_backend(name: str, target: str | type[Communicator], *, override: bool = False) -> None:
    """Register ``target`` (a class or a lazy ``'module:ClassName'`` path) under ``name``."""
    if name in _BACKENDS and not override:
        raise ValueError(f'communicator backend {name!r} is already registered')
    _BACKENDS[name] = target


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
        target = getattr(importlib.import_module(module_name), attr)
        _BACKENDS[name] = target
    return target


def create_communicator(
    rendezvous: Rendezvous,
    rank: int,
    world_size: int,
    device: torch.device | str,
    backend: str | None = None,
    **options,
) -> Communicator:
    """Build a standalone communication group. Collective over all ``world_size`` ranks.

    Args:
        rendezvous: Address rank 0 hosts the rendezvous store on; identical on every rank.
        rank: This process's rank inside the group (``0 .. world_size - 1``), not its global rank.
        world_size: Number of processes in the group; identical on every rank.
        device: Device the communicator binds to; tensors passed to it must live there.
        backend: Backend name; ``None`` lets the current platform decide.
        **options: Backend-specific settings; a backend ignores the ones it does not know.
    """
    return get_communicator_cls(backend).create(rendezvous, rank, world_size, device, **options)
