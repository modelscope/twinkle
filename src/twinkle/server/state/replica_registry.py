# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-backed registry of replica capacity and liveness.

Capacity and ``last_seen`` use separate keys so sampler liveness does not alter
the model-capacity data shape.
"""
from __future__ import annotations

import time

from .backend.base import StateBackend

REPLICA_PREFIX = 'replica::'
_MAX_LORAS_SUFFIX = '::max_loras'
_LAST_SEEN_SUFFIX = '::last_seen'


def _make_key(replica_id: str) -> str:
    return f'{REPLICA_PREFIX}{replica_id}{_MAX_LORAS_SUFFIX}'


def _last_seen_key(replica_id: str) -> str:
    return f'{REPLICA_PREFIX}{replica_id}{_LAST_SEEN_SUFFIX}'


def _replica_id_from_key(key: str) -> str | None:
    if not key.startswith(REPLICA_PREFIX) or not key.endswith(_MAX_LORAS_SUFFIX):
        return None
    return key[len(REPLICA_PREFIX):-len(_MAX_LORAS_SUFFIX)]


class ReplicaRegistry:
    """Read/write replica capacity and liveness through the shared backend."""

    def __init__(self, backend: StateBackend) -> None:
        self._backend = backend

    async def register(self, replica_id: str, max_loras: int) -> None:
        """Store / overwrite the declared LoRA capacity for ``replica_id``."""
        await self._backend.set(_make_key(replica_id), int(max_loras))

    async def unregister(self, replica_id: str) -> None:
        """Remove the capacity entry for ``replica_id`` (idempotent)."""
        await self._backend.delete(_make_key(replica_id))
        await self._backend.delete(_last_seen_key(replica_id))

    async def touch_last_seen(self, replica_id: str) -> None:
        """Refresh the replica's liveness timestamp (separate key from max_loras)."""
        await self._backend.set(_last_seen_key(replica_id), time.time())

    async def get_all_last_seen(self) -> dict[str, float]:
        """Return every replica's last-seen timestamp."""
        keys = await self._backend.keys(f'{REPLICA_PREFIX}*{_LAST_SEEN_SUFFIX}')
        out: dict[str, float] = {}
        for key in keys:
            if not key.startswith(REPLICA_PREFIX) or not key.endswith(_LAST_SEEN_SUFFIX):
                continue
            rid = key[len(REPLICA_PREFIX):-len(_LAST_SEEN_SUFFIX)]
            value = await self._backend.get(key)
            try:
                out[rid] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    async def get_all(self) -> dict[str, int]:
        """Return every registered replica's declared capacity."""
        keys = await self._backend.keys(f'{REPLICA_PREFIX}*{_MAX_LORAS_SUFFIX}')
        out: dict[str, int] = {}
        for key in keys:
            rid = _replica_id_from_key(key)
            if rid is None:
                continue
            value = await self._backend.get(key)
            try:
                out[rid] = int(value)
            except (TypeError, ValueError):
                continue
        return out
