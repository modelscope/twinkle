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
    """

    def __init__(self, expiration_timeout: float, per_token_model_limit: int = 30) -> None:
        super().__init__(expiration_timeout)
        self._per_token_model_limit = per_token_model_limit
        # token -> set of model_ids owned by that token
        self._token_models: dict[str, set[str]] = {}

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
        self._store[model_id] = record

    def remove(self, model_id: str) -> bool:
        """Remove a record by ID and clean up token ownership.

        Returns:
            True if the record existed and was removed, False otherwise.
        """
        record = self._store.pop(model_id, None)
        if record is None:
            return False
        token = record.token
        if token and token in self._token_models:
            self._token_models[token].discard(model_id)
            if not self._token_models[token]:
                del self._token_models[token]
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
            token = record.token
            if token and token in self._token_models:
                self._token_models[token].discard(model_id)
                if not self._token_models[token]:
                    del self._token_models[token]

        return len(expired_ids)
