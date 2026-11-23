# Copyright (c) ModelScope Contributors. All rights reserved.
"""Current, read-only operator views published by the cleanup leader."""

import math
import time
from datetime import UTC, datetime

from twinkle.server.telemetry.identity import user_fingerprint
from twinkle.utils.logger import get_logger
from .backend.base import StateBackend
from .model_manager import ModelManager
from .sampling_manager import SamplingSessionManager
from .session_manager import SessionManager
from .training_records import TERMINAL_TRAINING

logger = get_logger()


class AdminIndex:
    def __init__(
        self,
        backend: StateBackend,
        sessions: SessionManager,
        models: ModelManager,
        sampling: SamplingSessionManager,
        *,
        expiration_timeout: float,
        interval: float,
    ) -> None:
        self._backend = backend
        self._sessions = sessions
        self._models = models
        self._sampling = sampling
        self._expiration_timeout = expiration_timeout
        self._ttl = max(1, math.ceil(interval * 3))

    async def publish(self) -> None:
        now = time.time()
        updated_at = datetime.fromtimestamp(now, UTC).isoformat()
        cutoff = now - self._expiration_timeout
        for name, collect in (
            ("sessions", self._session_rows),
            ("trainings", self._training_rows),
            ("sampling", self._sampling_rows),
        ):
            try:
                rows = await collect(cutoff)
                for row in rows:
                    row["updated_at"] = updated_at
                await self._backend.set(
                    f"admin:index:{name}",
                    {
                        "updated_at": updated_at,
                        "rows": rows or [{"updated_at": updated_at, "state": "EMPTY"}],
                    },
                    ttl=self._ttl,
                )
            except Exception as exc:
                # The existing snapshot expires if this source stays unavailable.
                logger.warning("Unable to publish admin snapshot %s (%s)", name, type(exc).__name__)

    async def _session_rows(self, cutoff: float) -> list[dict]:
        sessions = await self._sessions.get_all()
        return [
            {
                "session_id": session_id,
                "user_id": record.user_id,
                "created_at": record.created_at,
                "last_heartbeat": datetime.fromtimestamp(record.last_heartbeat, UTC).isoformat(),
            }
            for session_id, record in sessions.items()
            if record.last_heartbeat >= cutoff
        ]

    async def _training_rows(self, _cutoff: float) -> list[dict]:
        records = await self._models.get_training_records()
        return [
            {
                "training_id": record.training_id,
                "session_id": record.session_id,
                "model_id": record.base_model,
                "user_id": user_fingerprint(record.token),
                "state": record.state,
                "created_at": record.created_at,
            }
            for record in records
            if record.state not in TERMINAL_TRAINING
        ]

    async def _sampling_rows(self, cutoff: float) -> list[dict]:
        sampling = await self._sampling.get_all()
        owners = list({record.session_id for record in sampling.values() if record.session_id})
        heartbeats = await self._sessions.get_last_heartbeats(owners)
        expired_owners = {owner for owner, heartbeat in heartbeats.items() if heartbeat is None or heartbeat < cutoff}
        return [
            {
                "sampling_session_id": sampling_id,
                "session_id": record.session_id,
                "model_id": record.base_model,
                "user_id": record.user_id,
                "state": "ACTIVE",
                "created_at": record.created_at,
            }
            for sampling_id, record in sampling.items()
            if not self._sampling.is_expired(record, cutoff, expired_owners)
        ]
