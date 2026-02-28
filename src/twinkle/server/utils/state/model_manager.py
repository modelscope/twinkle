# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from .base import BaseManager
from .models import ModelRecord


class ModelManager(BaseManager[ModelRecord]):
    """
    Manages registered models.

    Expiry is based on `created_at`.  A model is also considered expired if
    its owning session has already been removed (cascade expiry).

    Enforces a per-token model limit across all model instances (server-global).

    Also tracks replica registrations so the router can query which replicas
    still have capacity (i.e. their loaded-model count < max_loras).
    """

    def __init__(self, expiration_timeout: float, per_token_model_limit: int = 30) -> None:
        super().__init__(expiration_timeout)
        self._per_token_model_limit = per_token_model_limit
        # token -> set of model_ids owned by that token
        self._token_models: dict[str, set[str]] = {}
        # replica_id -> set of model_ids currently loaded on that replica
        self._replica_models: dict[str, set[str]] = {}
        # replica_id -> max_loras limit declared at registration time
        self._replica_max_loras: dict[str, int] = {}

    # ----- Replica Registration -----

    def register_replica(self, replica_id: str, max_loras: int) -> None:
        """Register a replica and its LoRA capacity.

        Args:
            replica_id: Unique identifier for the replica.
            max_loras: Maximum number of LoRA adapters the replica can hold.
        """
        self._replica_max_loras[replica_id] = max_loras
        self._replica_models.setdefault(replica_id, set())

    def unregister_replica(self, replica_id: str) -> None:
        """Remove a replica from the registry.

        Any model associations for this replica are also cleared.

        Args:
            replica_id: Unique identifier for the replica to remove.
        """
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

    def add(self, model_id: str, record: ModelRecord) -> None:
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
        self._token_models.setdefault(token, set()).add(model_id)
        if record.replica_id is not None:
            self._replica_models.setdefault(record.replica_id, set()).add(model_id)
        self._store[model_id] = record

    def remove(self, model_id: str) -> bool:
        """Remove a record by ID and clean up token and replica ownership.

        Returns:
            True if the record existed and was removed, False otherwise.
        """
        record = self._store.pop(model_id, None)
        if record is None:
            return False
        self._cleanup_ownership(model_id, record)
        return True

    # ----- Cleanup -----

    def cleanup_expired(self, cutoff_time: float, expired_session_ids: list[str] | None = None) -> int:
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
        expired_ids = []

        for model_id, record in self._store.items():
            # Cascade: owner session was expired
            if record.session_id and record.session_id in session_set:
                expired_ids.append(model_id)
                continue
            # Own age
            created_at = self._parse_timestamp(record.created_at)
            if created_at < cutoff_time:
                expired_ids.append(model_id)

        for model_id in expired_ids:
            record = self._store.pop(model_id)
            self._cleanup_ownership(model_id, record)

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
