# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource counts and current Redis views published by the cleanup leader."""

import asyncio
import math
import time
from datetime import UTC, datetime

from twinkle_server.identity import tenant_fingerprint
from twinkle_server.telemetry import MetricsRegistry
from twinkle.utils.logger import get_logger
from .backend.base import StateBackend
from .future_manager import FutureManager
from .model_manager import ModelManager
from .sampling_manager import SamplingSessionManager
from .session_manager import SessionManager
from .training_records import TERMINAL_TRAINING

logger = get_logger()


class ResourceSnapshotPublisher:
    def __init__(
        self,
        backend: StateBackend,
        sessions: SessionManager,
        models: ModelManager,
        sampling: SamplingSessionManager,
        futures: FutureManager,
        *,
        publish_details: bool,
        expiration_timeout: float,
        interval: float,
    ) -> None:
        self._backend = backend
        self._sessions = sessions
        self._models = models
        self._sampling = sampling
        self._expiration_timeout = expiration_timeout
        self._futures = futures
        self._interval = interval
        self._publish_details = publish_details
        self._ttl = max(1, math.ceil(interval * 3))
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        MetricsRegistry.get().clear_resource_counts()

    async def _publish_loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self._publish_once()

    async def _publish_once(self) -> None:
        registry = MetricsRegistry.get()
        now = time.time()
        updated_at = datetime.fromtimestamp(now, UTC).isoformat()
        cutoff = now - self._expiration_timeout
        for gauge, manager, name, collect in (
            ("active_sessions", self._sessions, "sessions", self._session_rows),
            ("active_models", self._models, "trainings", self._training_rows),
            ("active_sampling_sessions", self._sampling, "sampling", self._sampling_rows),
            ("active_futures", self._futures, None, None),
        ):
            try:
                if not self._publish_details or collect is None:
                    registry.set_resource_count(gauge, await manager.count())
                    continue
                ids = await manager.keys()
                registry.set_resource_count(gauge, len(ids))
                rows = await collect(ids, cutoff)
                await self._backend.set(
                    f"admin:index:{name}",
                    {"updated_at": updated_at, "row_count": len(rows), "rows": rows},
                    ttl=self._ttl,
                )
            except Exception as exc:
                # A failed update leaves the prior view available until its TTL expires.
                logger.warning("Unable to refresh resource view %s (%s)", gauge, type(exc).__name__)

    async def _session_rows(self, ids: list[str], cutoff: float) -> list[dict]:
        sessions = await self._sessions.get_all(resource_ids=ids)
        return [
            {
                "session_id": session_id,
                "tenant_fingerprint": record.user_id or "unassigned",
                "created_at": record.created_at,
                "last_heartbeat": datetime.fromtimestamp(record.last_heartbeat, UTC).isoformat(),
            }
            for session_id, record in sessions.items()
            if record.last_heartbeat >= cutoff
        ]

    async def _training_rows(self, ids: list[str], _cutoff: float) -> list[dict]:
        records = await self._models.get_training_records(model_ids=ids)
        return [
            {
                "training_id": record.training_id,
                "session_id": record.session_id,
                "base_model": record.base_model,
                "tenant_fingerprint": tenant_fingerprint(record.token),
                "state": record.state,
                "created_at": record.created_at,
            }
            for record in records
            if record.state not in TERMINAL_TRAINING
        ]

    async def _sampling_rows(self, ids: list[str], cutoff: float) -> list[dict]:
        sampling = await self._sampling.get_all(resource_ids=ids)
        owners = list({record.session_id for record in sampling.values() if record.session_id})
        heartbeats = await self._sessions.get_last_heartbeats(owners)
        expired_owners = {owner for owner, heartbeat in heartbeats.items() if heartbeat is None or heartbeat < cutoff}
        return [
            {
                "sampling_session_id": sampling_id,
                "session_id": record.session_id,
                "base_model": record.base_model,
                "tenant_fingerprint": record.user_id or "unassigned",
                "state": "ACTIVE",
                "created_at": record.created_at,
            }
            for sampling_id, record in sampling.items()
            if not self._sampling.is_expired(record, cutoff, expired_owners)
        ]
