# Copyright (c) ModelScope Contributors. All rights reserved.
"""Immutable client identity and the compatibility default-transport registry."""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ClientTransport

TWINKLE_SERVER_URL = os.environ.get('TWINKLE_SERVER_URL', 'http://127.0.0.1:8000')
TWINKLE_SERVER_TOKEN = os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN')


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip('/')
    return base_url if base_url.endswith('/api/v1') else f'{base_url}/api/v1'


@dataclass(frozen=True, slots=True)
class ClientContext:
    """A resolved request identity captured by one transport."""

    base_url: str
    api_key: str
    session_id: str | None = None
    routing_id: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'base_url', _normalize_base_url(self.base_url))
        if not self.routing_id:
            object.__setattr__(self, 'routing_id', uuid.uuid4().hex)


_default_lock = threading.RLock()
_default_transport: ClientTransport | None = None


def _new_env_transport() -> ClientTransport:
    from .client import ClientTransport
    return ClientTransport(
        ClientContext(
            base_url=os.environ.get('TWINKLE_SERVER_URL', TWINKLE_SERVER_URL),
            api_key=os.environ.get('TWINKLE_SERVER_TOKEN', TWINKLE_SERVER_TOKEN),
        ))


def capture_transport(explicit: ClientTransport | None = None) -> ClientTransport:
    """Return an explicit transport or capture the current compatibility default."""
    global _default_transport
    if explicit is not None:
        if explicit.closed:
            raise RuntimeError('Cannot capture a closed ClientTransport')
        return explicit
    with _default_lock:
        if _default_transport is None or _default_transport.closed:
            _default_transport = _new_env_transport()
        return _default_transport


def set_default_transport(transport: ClientTransport) -> None:
    if transport.closed:
        raise RuntimeError('Cannot register a closed ClientTransport')
    global _default_transport
    with _default_lock:
        _default_transport = transport


def clear_default_transport(transport: ClientTransport) -> None:
    global _default_transport
    with _default_lock:
        if _default_transport is transport:
            _default_transport = None


# Private compatibility seam for the deferred Tinker monkey patch. New Twinkle code
# must use ClientTransport directly; these names are intentionally not re-exported.
def get_api_key() -> str:
    return capture_transport().context.api_key


def get_request_id() -> str:
    return capture_transport().context.routing_id
