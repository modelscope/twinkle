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

from .backend.base import StateBackend
from .base import BaseManager
from .models import ModelRecord
from .replica_registry import ReplicaRegistry


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

    async def rebuild_indexes(self) -> None:
        """Compatibility shim — indexes are now derived from the backend per call."""
        return None

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
        """Store a record, enforcing the per-token model limit.

        Raises:
            RuntimeError: when adding ``record`` would exceed
                ``per_token_model_limit`` for ``record.token``.
        """
        token = record.token
        current = await self._count_models_for_token(token)
        if current >= self._per_token_model_limit:
            raise RuntimeError(f'Model limit exceeded: {current}/{self._per_token_model_limit} models')
        await super().add(model_id, record)

    async def remove(self, model_id: str) -> bool:
        """Remove a record by ID."""
        record = await self.get(model_id)
        if record is None:
            return False
        await super().remove(model_id)
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
            await self.remove(model_id)
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
