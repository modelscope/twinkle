# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .backend.base import StateBackend
from .base import BaseManager
from .models import ModelRecord


class ModelManager(BaseManager[ModelRecord]):
    """Manages registered models.

    Expiry is based on `created_at`.  A model is also considered expired if
    its owning session has already been removed (cascade expiry).

    Enforces a per-token model limit across all model instances (server-global).

    Also tracks replica registrations so the router can query which replicas
    still have capacity (i.e. their loaded-model count < max_loras).

    Uses a **hybrid mode**: primary records (ModelRecord) are persisted in the
    StateBackend, while derived indexes are kept in memory for fast lookups.
    On startup, `rebuild_indexes()` loads all records and rebuilds the indexes.
    """

    def __init__(self, backend: StateBackend, expiration_timeout: float, per_token_model_limit: int = 30) -> None:
        super().__init__(backend, "model::", ModelRecord, expiration_timeout)
        self._per_token_model_limit = per_token_model_limit
        # token -> set of model_ids owned by that token
        self._token_models: dict[str, set[str]] = {}
        # replica_id -> set of model_ids currently loaded on that replica
        self._replica_models: dict[str, set[str]] = {}
        # replica_id -> max_loras limit declared at registration time
        self._replica_max_loras: dict[str, int] = {}

    # ----- Index Rebuild -----

    async def rebuild_indexes(self) -> None:
        """Rebuild in-memory indexes from all records in the backend.

        Should be called once after startup (e.g. in ServerState.start_cleanup_task).
        """
        all_records = await self.get_all()
        self._token_models.clear()
        self._replica_models.clear()
        for model_id, record in all_records.items():
            token = record.token
            self._token_models.setdefault(token, set()).add(model_id)
            if record.replica_id is not None:
                self._replica_models.setdefault(record.replica_id, set()).add(model_id)

    # ----- Capacity Info -----

    def get_capacity_info(self) -> dict[str, int]:
        """Return global LoRA capacity across all registered replicas.

        Returns:
            Dict containing 'max_loras', 'used_loras', and 'free_loras'.
        """
        total_max_loras = sum(self._replica_max_loras.values())
        total_used_loras = sum(len(self._replica_models.get(rid, set())) for rid in self._replica_max_loras.keys())
        return {
            'max_loras': total_max_loras,
            'used_loras': total_used_loras,
            'free_loras': max(0, total_max_loras - total_used_loras),
        }

    # ----- Replica Registration -----

    def register_replica(self, replica_id: str, max_loras: int) -> None:
        """Register a replica and its LoRA capacity.

        Args:
            replica_id: Unique identifier for the replica.
            max_loras: Maximum number of LoRA adapters the replica can hold.
        """
        self._replica_max_loras[replica_id] = max_loras
        self._replica_models.setdefault(replica_id, set())

    async def unregister_replica(self, replica_id: str) -> None:
        """Remove a replica from the registry.

        Any model associations for this replica are also cleared from both
        the backend and the in-memory indexes.

        Args:
            replica_id: Unique identifier for the replica to remove.
        """
        # Remove models associated with this replica
        model_ids = list(self._replica_models.get(replica_id, set()))
        for model_id in model_ids:
            await self.remove(model_id)
        self._replica_max_loras.pop(replica_id, None)
        self._replica_models.pop(replica_id, None)

    def get_available_replica_ids(self, candidate_ids: list[str]) -> list[str]:
        """Return the subset of candidate replica IDs that still have capacity.

        A replica has capacity when its current loaded-model count is strictly
        less than its declared ``max_loras``.  Replicas that are not registered
        (unknown to this manager) are included as-is (conservative fallback).

        Args:
            candidate_ids: Replica IDs to evaluate.

        Returns:
            Filtered list preserving the original order.
        """
        available = []
        for rid in candidate_ids:
            max_loras = self._replica_max_loras.get(rid)
            if max_loras is None:
                # Unknown replica – include conservatively
                available.append(rid)
                continue
            current = len(self._replica_models.get(rid, set()))
            if current < max_loras:
                available.append(rid)
        return available

    # ----- CRUD -----

    async def add(self, model_id: str, record: ModelRecord) -> None:
        """Store a record under the given ID.

        Args:
            model_id: Unique identifier for the model.
            record: ModelRecord to store.

        Raises:
            RuntimeError: If the token has reached per_token_model_limit.
        """
        token = record.token
        current_ids = self._token_models.get(token, set())
        if len(current_ids) >= self._per_token_model_limit:
            raise RuntimeError(f'Model limit exceeded: '
                               f'{len(current_ids)}/{self._per_token_model_limit} models')
        # Persist to backend
        await super().add(model_id, record)
        # Update in-memory indexes
        self._token_models.setdefault(token, set()).add(model_id)
        if record.replica_id is not None:
            self._replica_models.setdefault(record.replica_id, set()).add(model_id)

    async def remove(self, model_id: str) -> bool:
        """Remove a record by ID and clean up token and replica ownership.

        Returns:
            True if the record existed and was removed, False otherwise.
        """
        # Get the record first for index cleanup
        record = await self.get(model_id)
        if record is None:
            return False
        # Remove from backend
        await super().remove(model_id)
        # Clean up in-memory indexes
        self._cleanup_ownership(model_id, record)
        return True

    # ----- Cleanup -----

    async def cleanup_expired(self, cutoff_time: float, expired_session_ids: list[str] | None = None, **kwargs) -> int:
        """Remove models that are older than cutoff_time, or whose owning
        session has already been expired.

        Args:
            cutoff_time: Unix timestamp threshold.
            expired_session_ids: Optional list of session IDs that have just
                been expired; any model belonging to one of these sessions will
                also be removed regardless of its own age.

        Returns:
            Number of models removed.
        """
        session_set = set(expired_session_ids or [])
        all_records = await self.get_all()
        expired_ids = []

        for model_id, record in all_records.items():
            # Cascade: owner session was expired
            if record.session_id and record.session_id in session_set:
                expired_ids.append(model_id)
                continue
            # Own age
            created_at = self._parse_timestamp(record.created_at)
            if created_at < cutoff_time:
                expired_ids.append(model_id)

        for model_id in expired_ids:
            await self.remove(model_id)

        return len(expired_ids)

    # ----- Internal helpers -----

    def _cleanup_ownership(self, model_id: str, record: ModelRecord) -> None:
        """Remove token and replica ownership entries for a model record.

        Args:
            model_id: The model ID being removed.
            record: The associated ModelRecord.
        """
        token = record.token
        if token and token in self._token_models:
            self._token_models[token].discard(model_id)
            if not self._token_models[token]:
                del self._token_models[token]
        if record.replica_id and record.replica_id in self._replica_models:
            self._replica_models[record.replica_id].discard(model_id)
