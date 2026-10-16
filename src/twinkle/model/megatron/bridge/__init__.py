# Copyright (c) ModelScope Contributors. All rights reserved.
"""Bridge backends: the abstraction over mcore-bridge / megatron-bridge.

``resolve_bridge_backend`` maps a backend NAME to an instance. The name (not a live object) is what
crosses process boundaries: twinkle's ``@remote_class`` re-instantiates only the decorated base
class inside each Ray worker and forwards the constructor kwargs, so a constructed backend cannot
ride along, but a plain string can. ``MegatronStrategy`` therefore accepts ``bridge_backend`` as
either a name string or an already-built instance (the latter for local/test injection).
"""
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
