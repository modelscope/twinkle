# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-backed registry of replica capacity.

Each entry persists to ``replica::<replica_id>::max_loras`` in the configured
:class:`StateBackend` (Redis or the actor-wrapped RayActorBackend), so every
Ray Serve worker sees one consistent view of the cluster's capacity even
though each worker holds its own ``ServerState`` instance.

The registry knows *only* about declared capacity. The current loaded-model
count is derived by querying the persisted ``model::*`` records directly —
nothing here caches that count, so concurrent writes from different workers
cannot drift into an inconsistent local index.
"""
from __future__ import annotations

from .backend.base import StateBackend

REPLICA_PREFIX = 'replica::'
_MAX_LORAS_SUFFIX = '::max_loras'


def _make_key(replica_id: str) -> str:
    return f'{REPLICA_PREFIX}{replica_id}{_MAX_LORAS_SUFFIX}'


def _replica_id_from_key(key: str) -> str | None:
    if not key.startswith(REPLICA_PREFIX) or not key.endswith(_MAX_LORAS_SUFFIX):
        return None
    return key[len(REPLICA_PREFIX):-len(_MAX_LORAS_SUFFIX)]


class ReplicaRegistry:
    """Read/write replica capacity through the shared :class:`StateBackend`."""

    def __init__(self, backend: StateBackend) -> None:
        self._backend = backend

    async def register(self, replica_id: str, max_loras: int) -> None:
        """Store / overwrite the declared LoRA capacity for ``replica_id``."""
        await self._backend.set(_make_key(replica_id), int(max_loras))

    async def unregister(self, replica_id: str) -> None:
        """Remove the capacity entry for ``replica_id`` (idempotent)."""
        await self._backend.delete(_make_key(replica_id))

    async def get_max_loras(self, replica_id: str) -> int | None:
        """Return the declared capacity, or ``None`` if the replica is unknown."""
        value = await self._backend.get(_make_key(replica_id))
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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
