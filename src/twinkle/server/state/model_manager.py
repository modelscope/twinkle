# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-backed model manager.

Every index this manager exposes — token → model count, replica → loaded model
count, replica capacity — is computed from the persisted ``model::*`` and
``replica::*`` records in the shared :class:`StateBackend`. There is no
in-process cache, so two workers connected to the same backend see one
consistent view of the cluster's model registry without an extra coordination
actor in front.
"""
from __future__ import annotations

import functools

from .backend.base import StateBackend
from .base import BaseManager
from .models import ModelRecord
from .replica_registry import ReplicaRegistry

# Per-token model-count keys live in their own keyspace so they are never
# picked up by ``model::*`` scans. The count is maintained atomically via
# ``StateBackend.update_atomic`` so two concurrent adds with the same token
# cannot both pass the limit check (the prior count-then-add race).
_TOKEN_COUNT_PREFIX = 'token_count::'


def _counter_delta_transform(existing: object, *, delta: int) -> int:
    """Atomic transform body for the per-token counter (module-level for pickling).

    Treats a missing/!int value as 0 and never lets the stored count go
    negative. Always returns an int, so ``update_atomic`` writes and returns
    the new value (it never no-ops here).
    """
    current = existing if isinstance(existing, int) else 0
    new = current + delta
    return new if new > 0 else 0


class ModelManager(BaseManager[ModelRecord]):
    """Manages registered models with backend-derived per-token / per-replica indexes.

    Expiry is based on ``created_at``. A model is also considered expired if
    its owning session has already been removed (cascade expiry).
    Enforces a per-token model limit across all model instances (server-global).
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float, per_token_model_limit: int = 30) -> None:
        super().__init__(backend, 'model::', ModelRecord, expiration_timeout)
        self._per_token_model_limit = per_token_model_limit
        self._replicas = ReplicaRegistry(backend)

    # ----- Index Rebuild -------------------------------------------------- #

    def _token_count_key(self, token: str) -> str:
        return f'{_TOKEN_COUNT_PREFIX}{token}'

    async def rebuild_indexes(self) -> None:
        """Rebuild the per-token model counters from the persisted ``model::*`` records.

        Called on cleanup-task start. The atomic ``token_count::<token>`` keys
        are derived data; rebuilding them from the authoritative model records
        keeps them correct after a restart or a crash that left them stale.
        """
        all_records = await self.get_all()
        counts: dict[str, int] = {}
        for record in all_records.values():
            if record.token:
                counts[record.token] = counts.get(record.token, 0) + 1

        # Reset any stale counters, then write the recomputed ones.
        stale_keys = await self._backend.keys(f'{_TOKEN_COUNT_PREFIX}*')
        for key in stale_keys:
            await self._backend.delete(key)
        for token, count in counts.items():
            await self._backend.set(self._token_count_key(token), count)

    # ----- Capacity ------------------------------------------------------- #

    async def get_capacity_info(self) -> dict[str, int]:
        """Return global LoRA capacity across all registered replicas."""
        replicas = await self._replicas.get_all()
        loaded_per_replica = await self._loaded_per_replica()
        total_max = sum(replicas.values())
        total_used = sum(loaded_per_replica.get(rid, 0) for rid in replicas)
        return {
            'max_loras': total_max,
            'used_loras': total_used,
            'free_loras': max(0, total_max - total_used),
        }

    # ----- Replica Registration ------------------------------------------ #

    async def register_replica(self, replica_id: str, max_loras: int) -> None:
        """Register a replica and its LoRA capacity in the shared backend."""
        await self._replicas.register(replica_id, max_loras)

    async def unregister_replica(self, replica_id: str) -> None:
        """Remove a replica's capacity entry and any models it owns."""
        loaded = await self._models_for_replica(replica_id)
        for model_id in loaded:
            await self.remove(model_id)
        await self._replicas.unregister(replica_id)

    async def get_available_replica_ids(self, candidate_ids: list[str]) -> list[str]:
        """Return the subset of ``candidate_ids`` that still have capacity.

        A replica has capacity when its persisted loaded-model count is strictly
        less than its declared ``max_loras``. Unknown replicas (no capacity row
        in the backend) are included conservatively — callers may not have
        registered every candidate up front.
        """
        if not candidate_ids:
            return []
        replicas = await self._replicas.get_all()
        loaded_per_replica = await self._loaded_per_replica(replica_filter=set(candidate_ids) | replicas.keys())
        available: list[str] = []
        for rid in candidate_ids:
            max_loras = replicas.get(rid)
            if max_loras is None:
                # Unknown replica — include conservatively.
                available.append(rid)
                continue
            if loaded_per_replica.get(rid, 0) < max_loras:
                available.append(rid)
        return available

    # ----- CRUD ----------------------------------------------------------- #

    async def add(self, model_id: str, record: ModelRecord) -> None:
        """Store a record, enforcing the per-token model limit atomically.

        The per-token count is incremented through ``update_atomic`` BEFORE the
        record is written, so two concurrent adds with the same token cannot
        both observe ``limit - 1`` and both succeed (the prior count-then-add
        race). If the increment would exceed the limit, it is rolled back and a
        ``RuntimeError`` is raised; if the record write fails, the increment is
        rolled back too so the counter never drifts above the real model count.

        Raises:
            RuntimeError: when adding ``record`` would exceed
                ``per_token_model_limit`` for ``record.token``.
        """
        token = record.token
        if not token:
            # No token → no per-token limit to enforce.
            await super().add(model_id, record)
            return

        key = self._token_count_key(token)
        # The transform always returns an int, so update_atomic always writes
        # and returns the new count (it never no-ops here).
        new_count = await self._backend.update_atomic(
            key,
            functools.partial(_counter_delta_transform, delta=1),
        )
        if new_count > self._per_token_model_limit:
            # Roll the speculative increment back and reject. ``new_count - 1``
            # is the count that was already present before this add.
            await self._backend.update_atomic(key, functools.partial(_counter_delta_transform, delta=-1))
            raise RuntimeError(f'Model limit exceeded: {new_count - 1}/{self._per_token_model_limit} models')

        try:
            await super().add(model_id, record)
        except Exception:
            # Keep the counter consistent with the persisted records.
            await self._backend.update_atomic(key, functools.partial(_counter_delta_transform, delta=-1))
            raise

    async def remove(self, model_id: str, *, _record: ModelRecord | None = None) -> bool:
        """Remove a record by ID, decrementing its owning token's counter.

        When the caller already holds the record (e.g. from a prior ``get_all``),
        pass it via ``_record`` to skip the redundant backend fetch.
        """
        record = _record or await self.get(model_id)
        if record is None:
            return False
        await super().remove(model_id)
        if record.token:
            await self._backend.update_atomic(
                self._token_count_key(record.token),
                functools.partial(_counter_delta_transform, delta=-1),
            )
        return True

    # ----- Cleanup -------------------------------------------------------- #

    async def cleanup_expired(self, cutoff_time: float, expired_session_ids: list[str] | None = None, **kwargs) -> int:
        """Remove models older than ``cutoff_time`` or whose owning session expired."""
        session_set = set(expired_session_ids or [])
        all_records = await self.get_all()
        expired_ids: list[str] = []
        for model_id, record in all_records.items():
            if record.session_id and record.session_id in session_set:
                expired_ids.append(model_id)
                continue
            created_at = self._parse_timestamp(record.created_at)
            if created_at < cutoff_time:
                expired_ids.append(model_id)
        for model_id in expired_ids:
            await self.remove(model_id, _record=all_records[model_id])
        return len(expired_ids)

    # ----- Backend-derived helpers --------------------------------------- #

    async def _count_models_for_token(self, token: str | None) -> int:
        if not token:
            return 0
        all_records = await self.get_all()
        return sum(1 for r in all_records.values() if r.token == token)

    async def _models_for_replica(self, replica_id: str) -> list[str]:
        all_records = await self.get_all()
        return [mid for mid, r in all_records.items() if r.replica_id == replica_id]

    async def _loaded_per_replica(self, replica_filter: set[str] | None = None) -> dict[str, int]:
        all_records = await self.get_all()
        counts: dict[str, int] = {}
        for record in all_records.values():
            rid = record.replica_id
            if rid is None:
                continue
            if replica_filter is not None and rid not in replica_filter:
                continue
            counts[rid] = counts.get(rid, 0) + 1
        return counts
