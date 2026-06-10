# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import functools
import re
import time
import uuid
from datetime import datetime
from typing import Any

from twinkle.server.config.persistence import PersistenceConfig
from twinkle.server.telemetry import MetricsRegistry
from twinkle.server.telemetry.correlation import (BASE_MODEL, MODEL_ID, REPLICA_ID, SAMPLING_SESSION_ID, SESSION_ID,
                                                  TOKEN_ID)
from twinkle.server.telemetry.tracing import traced_operation
from twinkle.utils.logger import get_logger
from .backend import StateBackend
from .backend.factory import create_backend
from .config_manager import ConfigManager
from .future_manager import FutureManager
from .model_manager import ModelManager
from .models import ModelRecord, SamplingSessionRecord, SessionRecord
from .sampling_manager import SamplingSessionManager
from .session_manager import SessionManager

logger = get_logger()

# ---------- Cleanup-leader election ------------------------------------------
#
# Every Ray Serve worker creates its own ``ServerState``; without coordination
# each one would run the periodic cleanup and metrics-publish loop, so a
# single Twinkle deployment would multiply the work and inflate every gauge
# by the worker count. We elect one leader per backend by racing for a TTL-
# scoped key inside the shared StateBackend: the winner runs cleanup +
# publishes metrics, the others stay quiet.

LEADER_KEY = 'cleanup_leader'  # actual backend key: '<key_prefix>cleanup_leader'
LEASE_TTL = 30  # seconds — leader loses the lease after this without a renew
LEASE_RENEW = 10  # seconds — must be < LEASE_TTL/2 so two missed renews still beat the TTL


def _renew_if_owner(current: str | None, *, owner: str) -> str | None:
    """``update_atomic`` transform: only re-write the lease if it is still mine."""
    if current == owner:
        return owner
    return None


class ServerState:
    """Unified server state management class.

    Composes five resource managers:

    - :class:`SessionManager` — client sessions
    - :class:`ModelManager` — registered models
    - :class:`SamplingSessionManager` — sampling sessions
    - :class:`FutureManager` — async task futures
    - :class:`ConfigManager` — key-value configuration

    Each Ray Serve worker owns one process-local instance, bound directly to a
    shared :class:`StateBackend`. The cleanup loop is started from the
    deployment's FastAPI ``lifespan`` startup hook and only runs in the worker
    that wins the cleanup-leader lease — see :meth:`_leader_loop`.
    """

    def __init__(
            self,
            backend: StateBackend | None = None,
            persistence_config: PersistenceConfig | None = None,
            expiration_timeout: float = 86400.0,  # 24 hours in seconds
            cleanup_interval: float = 3600.0,  # 1 hour in seconds
            per_token_model_limit: int = 30,
            metrics_update_interval: float = 15.0) -> None:
        if backend is not None:
            self._backend: StateBackend = backend
        else:
            self._backend = create_backend(persistence_config)
        self._session_mgr = SessionManager(self._backend, expiration_timeout)
        self._model_mgr = ModelManager(self._backend, expiration_timeout, per_token_model_limit)
        self._sampling_mgr = SamplingSessionManager(self._backend, expiration_timeout)
        self._future_mgr = FutureManager(self._backend, expiration_timeout)
        self._config_mgr = ConfigManager(self._backend)

        self.expiration_timeout = expiration_timeout
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_running = False

        # Leader election + metrics-publish loop state. ``metrics_update_interval``
        # is a typed parameter (a misspelled key now fails loudly rather than
        # being silently ignored); it controls how often the leader pushes counts
        # into the MetricsRegistry cache.
        self._leader_id = uuid.uuid4().hex
        self._is_leader = False
        self._leader_task: asyncio.Task | None = None
        self._leader_running = False
        self._metrics_publish_task: asyncio.Task | None = None
        self._metrics_publish_running = False
        self._metrics_update_interval: float = float(metrics_update_interval)

    async def get_capacity_info(self) -> dict[str, int]:
        return await self._model_mgr.get_capacity_info()

    # ----- Session Management -----

    async def create_session(self, payload: dict[str, Any]) -> str:
        """Create a new session with the given payload.

        Args:
            payload: Session configuration containing optional session_id, tags, etc.

        Returns:
            The session_id for the created session.
        """
        session_id = payload.get('session_id') or f'session_{uuid.uuid4().hex}'
        with traced_operation(
                'server_state.create_session',
                attrs={SESSION_ID: session_id},
        ):
            record = SessionRecord(
                tags=list(payload.get('tags') or []),
                user_metadata=payload.get('user_metadata') or {},
                sdk_version=payload.get('sdk_version'),
            )
            await self._session_mgr.add(session_id, record)
            return session_id

    async def touch_session(self, session_id: str) -> bool:
        """Update session heartbeat timestamp.

        Returns:
            True if the session exists and was touched, False otherwise.
        """
        return await self._session_mgr.touch(session_id)

    async def get_session_last_heartbeat(self, session_id: str) -> float | None:
        """Get the last heartbeat timestamp for a session.

        Returns:
            Last heartbeat timestamp, or None if the session does not exist.
        """
        return await self._session_mgr.get_last_heartbeat(session_id)

    # ----- Model Registration -----

    async def register_model(self,
                             payload: dict[str, Any],
                             token: str,
                             model_id: str | None = None,
                             replica_id: str | None = None,
                             session_id: str | None = None) -> str:
        """Register a new model with the server state.

        Args:
            payload: Model configuration containing base_model, lora_config, etc.
            token: User token that owns this model. Required.
            model_id: Optional explicit model_id; otherwise auto-generated.
            replica_id: Optional replica that is hosting this model.
            session_id: Optional owning session; enables cascade cleanup when
                the session expires. Falls back to ``payload['session_id']``.

        Returns:
            The model_id for the registered model.
        """
        _time = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_id: str = model_id or payload.get(
            'model_id') or f"{_time}-{payload.get('base_model', 'model')}-{uuid.uuid4().hex[:8]}"
        _model_id = re.sub(r'[^\w\-]', '_', _model_id)

        with traced_operation(
                'server_state.register_model',
                attrs={
                    MODEL_ID: _model_id,
                    BASE_MODEL: payload.get('base_model'),
                    REPLICA_ID: replica_id,
                    TOKEN_ID: token,
                    SESSION_ID: session_id or payload.get('session_id'),
                },
        ):
            record = ModelRecord(
                session_id=session_id or payload.get('session_id'),
                model_seq_id=payload.get('model_seq_id'),
                base_model=payload.get('base_model'),
                user_metadata=payload.get('user_metadata') or {},
                lora_config=payload.get('lora_config'),
                token=token,
                replica_id=replica_id,
            )
            await self._model_mgr.add(_model_id, record)
            return _model_id

    async def unload_model(self, model_id: str) -> bool:
        """Remove a model from the registry.

        Returns:
            True if the model was found and removed, False otherwise.
        """
        return await self._model_mgr.remove(model_id)

    async def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
        """Get metadata for a registered model as a plain dict."""
        record = await self._model_mgr.get(model_id)
        return record.model_dump() if record is not None else None

    # ----- Replica Management -----

    async def register_replica(self, replica_id: str, max_loras: int) -> None:
        """Register a replica and its LoRA capacity.

        Args:
            replica_id: Unique identifier for the replica.
            max_loras: Maximum number of LoRA adapters the replica can hold.
        """
        with traced_operation(
                'server_state.register_replica',
                attrs={REPLICA_ID: replica_id},
        ):
            await self._model_mgr.register_replica(replica_id, max_loras)

    async def unregister_replica(self, replica_id: str) -> None:
        """Remove a replica from the registry.

        Args:
            replica_id: Unique identifier for the replica to remove.
        """
        await self._model_mgr.unregister_replica(replica_id)

    async def get_available_replica_ids(self, candidate_ids: list[str]) -> list[str]:
        """Return candidate replica IDs that have not reached their max_loras limit.

        Args:
            candidate_ids: Replica IDs to evaluate.

        Returns:
            Filtered list of replica IDs with remaining capacity.
        """
        return await self._model_mgr.get_available_replica_ids(candidate_ids)

    # ----- Sampling Session Management -----

    async def create_sampling_session(self, payload: dict[str, Any], sampling_session_id: str | None = None) -> str:
        """Create a new sampling session.

        Args:
            payload: Session configuration.
            sampling_session_id: Optional explicit ID.

        Returns:
            The sampling_session_id.
        """
        _sampling_session_id: str = sampling_session_id or payload.get(
            'sampling_session_id') or f'sampling_{uuid.uuid4().hex}'
        with traced_operation(
                'server_state.create_sampling_session',
                attrs={
                    SAMPLING_SESSION_ID: _sampling_session_id,
                    SESSION_ID: payload.get('session_id'),
                    BASE_MODEL: payload.get('base_model'),
                },
        ):
            record = SamplingSessionRecord(
                session_id=payload.get('session_id'),
                seq_id=payload.get('sampling_session_seq_id'),
                base_model=payload.get('base_model'),
                model_path=payload.get('model_path'),
            )
            await self._sampling_mgr.add(_sampling_session_id, record)
            return _sampling_session_id

    async def get_sampling_session(self, sampling_session_id: str) -> dict[str, Any] | None:
        """Get a sampling session by ID as a plain dict."""
        record = await self._sampling_mgr.get(sampling_session_id)
        return record.model_dump() if record is not None else None

    # ----- Future Management -----

    async def get_future(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve a stored future result as a plain dict."""
        record = await self._future_mgr.get(request_id)
        return record.model_dump() if record is not None else None

    async def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: str | None,
        reason: str | None = None,
        result: Any = None,
        queue_state: str | None = None,
        queue_state_reason: str | None = None,
    ) -> None:
        """Store task status with optional result.

        Supports the full task lifecycle:
        - PENDING: Task created, waiting to be processed
        - QUEUED: Task in queue waiting for execution
        - RUNNING: Task currently executing
        - COMPLETED: Task completed successfully (result required)
        - FAILED: Task failed with error (result contains error payload)
        - RATE_LIMITED: Task rejected due to rate limiting (reason required)

        Args:
            request_id: Unique identifier for the request.
            status: Task status string (pending/queued/running/completed/failed/rate_limited).
            model_id: Optional associated model_id.
            reason: Optional reason string (used for rate_limited status).
            result: Optional result data (used for completed/failed status).
            queue_state: Optional queue state for tinker client (active/paused_rate_limit/paused_capacity).
            queue_state_reason: Optional reason for the queue state.
        """
        await self._future_mgr.store_status(
            request_id=request_id,
            status=status,
            model_id=model_id,
            reason=reason,
            result=result,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
        )

    # ----- Configuration Management -----

    async def add_config(self, key: str, value: Any) -> None:
        """Add or overwrite a configuration value."""
        await self._config_mgr.add(key, value)

    async def add_or_get_config(self, key: str, value: Any) -> Any:
        """Add a config value if absent; otherwise return the existing value."""
        return await self._config_mgr.add_or_get(key, value)

    async def get_config(self, key: str) -> Any | None:
        """Return the configuration value for key, or None."""
        return await self._config_mgr.get(key)

    async def pop_config(self, key: str) -> Any | None:
        """Remove and return the configuration value for key, or None."""
        return await self._config_mgr.pop(key)

    async def clear_config(self) -> None:
        """Remove all configuration entries."""
        await self._config_mgr.clear()

    async def count_config(self) -> int:
        """Return the number of stored configuration entries."""
        return await self._config_mgr.count()

    # ----- Resource Cleanup -----

    async def cleanup_expired_resources(self) -> dict[str, int]:
        """Clean up expired sessions, models, sampling_sessions, and futures.

        Sessions expire based on last_heartbeat (or created_at).  Models and
        sampling sessions are also cascade-expired when their owning session
        expires.  Futures expire based on updated_at (or created_at).

        Returns:
            Dict with counts of cleaned up resources by type.
        """
        current_time = time.time()
        cutoff_time = current_time - self.expiration_timeout

        # Determine expired sessions and remove them in a SINGLE pass, then
        # cascade the SAME set to dependent resources. Using one authoritative
        # set (rather than a separate expiry scan followed by a second scan in
        # cleanup) closes the TOCTOU window where a session touched mid-cleanup
        # could survive removal while its children were cascade-deleted.
        expired_session_ids, sessions_removed = await self._session_mgr.collect_and_remove_expired(cutoff_time)

        models_removed = await self._model_mgr.cleanup_expired(cutoff_time, expired_session_ids=expired_session_ids)
        samplings_removed = await self._sampling_mgr.cleanup_expired(
            cutoff_time, expired_session_ids=expired_session_ids)
        futures_removed = await self._future_mgr.cleanup_expired(cutoff_time)

        return {
            'sessions': sessions_removed,
            'models': models_removed,
            'sampling_sessions': samplings_removed,
            'futures': futures_removed,
        }

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired resources.

        Gated by leader election — non-leader workers skip the actual cleanup
        so the same backend isn't swept 4× by 4 deployment workers.
        """
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if not self._is_leader:
                    continue
                stats = await self.cleanup_expired_resources()
                if any(stats.values()):
                    logger.debug(f'[ServerState Cleanup] Removed expired resources: {stats}')
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f'[ServerState Cleanup] Error during cleanup: {e}')
                continue

    # ----- Leader election + metrics publish -----

    async def _leader_loop(self) -> None:
        """Acquire and renew the cleanup-leader lease every LEASE_RENEW seconds."""
        await self._try_acquire_or_renew()  # Race for leadership at startup
        while self._leader_running:
            try:
                await asyncio.sleep(LEASE_RENEW)
                await self._try_acquire_or_renew()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f'[ServerState Leader] renew error: {e}')
                continue

    async def _try_acquire_or_renew(self) -> None:
        was_leader = self._is_leader
        try:
            if self._is_leader:
                val = await self._backend.update_atomic(
                    LEADER_KEY,
                    functools.partial(_renew_if_owner, owner=self._leader_id),
                    ttl=LEASE_TTL,
                )
                self._is_leader = (val == self._leader_id)
            else:
                self._is_leader = await self._backend.set_nx(LEADER_KEY, self._leader_id, ttl=LEASE_TTL)
        except Exception as e:
            logger.warning(f'[ServerState Leader] backend error during election: {e}')
            self._is_leader = False
            if was_leader:
                # Our renewal failed but our lease value may still be sitting in
                # the backend, so a plain ``set_nx`` would keep returning False
                # for up to LEASE_TTL and leadership would stall unclaimed.
                # Best-effort delete ONLY when we were the leader (never steal a
                # lease another replica legitimately holds), swallowing errors so
                # a delete failure cannot escape the election loop. The next tick
                # can then re-acquire immediately.
                try:
                    await self._backend.delete(LEADER_KEY)
                except Exception:
                    pass

        if self._is_leader and not was_leader:
            await self._on_become_leader()
        elif not self._is_leader and was_leader:
            await self._on_lose_leader()

    async def _on_become_leader(self) -> None:
        logger.info(f'[ServerState] became cleanup leader (id={self._leader_id[:8]})')
        # Start pushing resource counts so the four ObservableGauges in the
        # MetricsRegistry have a single source of truth across deployments.
        if self._metrics_publish_task is None or self._metrics_publish_task.done():
            self._metrics_publish_running = True
            self._metrics_publish_task = asyncio.create_task(self._metrics_publish_loop())

    async def _on_lose_leader(self) -> None:
        logger.warning(f'[ServerState] lost cleanup leadership (id={self._leader_id[:8]})')
        self._metrics_publish_running = False
        if self._metrics_publish_task is not None:
            self._metrics_publish_task.cancel()
            try:
                await self._metrics_publish_task
            except (asyncio.CancelledError, Exception):
                pass
            self._metrics_publish_task = None
        # Clear this worker's resource-gauge cache after cancelling the publish
        # task. Across replicas the old leader's MetricsRegistry cache lives in
        # a different process, and after handover its publish loop is cancelled
        # and never overwrites the cache again — so without this zeroing the
        # stale worker would keep emitting its last counts forever. The new
        # leader publishes the authoritative counts from its own process.
        MetricsRegistry.get().clear_resource_counts()

    async def _metrics_publish_loop(self) -> None:
        """Push resource counts into the MetricsRegistry cache every N seconds.

        Only runs while this ``ServerState`` holds the cleanup-leader lease.
        The ObservableGauges registered by :class:`MetricsRegistry` read the
        cache at OTEL export time and report whatever was pushed last.
        """
        registry = MetricsRegistry.get()
        sources = (
            ('active_sessions', self._session_mgr),
            ('active_models', self._model_mgr),
            ('active_sampling_sessions', self._sampling_mgr),
            ('active_futures', self._future_mgr),
        )
        while self._metrics_publish_running:
            try:
                await asyncio.sleep(self._metrics_update_interval)
                if not self._is_leader:
                    continue
                for name, mgr in sources:
                    registry.set_resource_count(name, await mgr.count())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'[ServerState] Error publishing metrics: {e}')
                continue

    async def start_cleanup_task(self) -> bool:
        """Start the background cleanup + leader-election tasks.

        Returns:
            True if tasks were started, False if already running.
        """
        if self._cleanup_running:
            return False
        # Rebuild in-memory indexes from backend data
        await self._rebuild_indexes()
        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._leader_running = True
        self._leader_task = asyncio.create_task(self._leader_loop())
        return True

    async def _rebuild_indexes(self) -> None:
        """Rebuild in-memory indexes from backend data after startup."""
        # Rebuild model indexes
        await self._model_mgr.rebuild_indexes()

    async def stop_cleanup_task(self) -> bool:
        """Stop the background cleanup + leader-election tasks.

        Returns:
            True if tasks were stopped, False if not running.
        """
        if not self._cleanup_running:
            return False
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        self._leader_running = False
        if self._leader_task:
            self._leader_task.cancel()
            self._leader_task = None
        if self._is_leader:
            # Release callback registration; the lease itself expires on its own
            # TTL — update_atomic can't express "atomic delete", so we accept a
            # short outage where the gauge reads 0 between leaders.
            await self._on_lose_leader()
            self._is_leader = False
        return True

    async def get_cleanup_stats(self) -> dict[str, Any]:
        """Get current cleanup configuration and resource counts.

        Returns:
            Dict with cleanup configuration and task status.
        """
        return {
            'expiration_timeout': self.expiration_timeout,
            'cleanup_interval': self.cleanup_interval,
            'cleanup_running': self._cleanup_running,
            'is_leader': self._is_leader,
            'leader_id': self._leader_id,
            'resource_counts': {
                'sessions': await self._session_mgr.count(),
                'models': await self._model_mgr.count(),
                'sampling_sessions': await self._sampling_mgr.count(),
                'futures': await self._future_mgr.count(),
            },
        }


# ---------------------------------------------------------------------------
# Per-process ServerState cache
# ---------------------------------------------------------------------------
#
# Each Ray Serve worker binds one ``ServerState`` instance to the shared
# ``StateBackend`` for the lifetime of the process — the cleanup loop and
# leader-election loop are started exactly once per worker (see
# ``start_cleanup_task``). Callers use ``actor_name`` as the cache key purely
# for per-process deduplication; cross-worker coordination happens inside the
# shared backend, not in this dict.

_PROCESS_STATE_CACHE: dict[str, ServerState] = {}


def get_server_state(actor_name: str = 'twinkle_server_state',
                     backend: StateBackend | None = None,
                     persistence_config: PersistenceConfig | None = None,
                     expiration_timeout: float = 86400.0,
                     cleanup_interval: float = 3600.0,
                     per_token_model_limit: int = 30,
                     metrics_update_interval: float = 15.0) -> ServerState:
    """Return a process-local :class:`ServerState` bound directly to the backend.

    Within one process the same ``actor_name`` returns the same cached instance
    so repeated callers share one ``ServerState`` and the cleanup loop is
    started exactly once. Cross-worker consistency comes from the shared
    :class:`StateBackend` rather than from any singleton in this process.

    Args:
        actor_name: Cache key for the per-process ``ServerState`` instance.
            The legacy parameter name is kept for call-site compatibility.
        backend: Optional :class:`StateBackend` to inject. When ``None`` a
            backend is built from ``persistence_config`` (or env vars) via
            :func:`create_backend`.
        persistence_config: Optional :class:`PersistenceConfig`. Accepted as a
            raw dict for YAML compatibility.
        expiration_timeout: Forwarded to :class:`ServerState`.
        cleanup_interval: Forwarded to :class:`ServerState`.
        per_token_model_limit: Forwarded to :class:`ServerState`.
        metrics_update_interval: Forwarded to :class:`ServerState`.
    """
    if isinstance(persistence_config, dict):
        persistence_config = PersistenceConfig(**persistence_config)

    if backend is None and persistence_config is None:
        persistence_config = PersistenceConfig.from_env()

    cached = _PROCESS_STATE_CACHE.get(actor_name)
    if cached is not None:
        return cached

    state = ServerState(
        backend=backend,
        persistence_config=persistence_config,
        expiration_timeout=expiration_timeout,
        cleanup_interval=cleanup_interval,
        per_token_model_limit=per_token_model_limit,
        metrics_update_interval=metrics_update_interval,
    )
    _PROCESS_STATE_CACHE[actor_name] = state
    # Cleanup task is started by the deployment's FastAPI ``lifespan`` hook
    # via ``await state.start_cleanup_task()`` — that's the single async
    # entry point each worker has, so we don't need any sync-context
    # detection here.
    return state


def reset_server_state_cache() -> None:
    """Clear the per-process ServerState cache.

    Test-only helper. Production code should never need to reset state across
    requests — workers reuse one instance for the lifetime of the process.
    """
    _PROCESS_STATE_CACHE.clear()
