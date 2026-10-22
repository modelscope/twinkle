# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Dict, Type

from .mcore import MCoreBridgeBackend
from .megatron_bridge import MegatronBridgeBackend
from .protocol import BridgeBackend

_BACKENDS: Dict[str, Type] = {
    MCoreBridgeBackend.backend_name: MCoreBridgeBackend,
    MegatronBridgeBackend.backend_name: MegatronBridgeBackend,
}


def resolve_bridge_backend(name: str) -> BridgeBackend:
    """Instantiate the backend registered under ``name``; fail loudly on an unknown one."""
    try:
        backend_cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(f'Unknown bridge_backend {name!r}. Supported: {sorted(_BACKENDS)}.') from None
    return backend_cls()


__all__ = ['BridgeBackend', 'MCoreBridgeBackend', 'MegatronBridgeBackend', 'resolve_bridge_backend']
