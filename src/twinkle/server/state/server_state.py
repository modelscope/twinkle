# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import functools
import math
import re
import time
import uuid
from datetime import datetime
from typing import Any

from twinkle.server.config.persistence import PersistenceConfig
from twinkle.server.exceptions import ResourceQuotaExceededError
from twinkle.server.telemetry.correlation import (BASE_MODEL, MODEL_ID, REPLICA_ID, SAMPLING_SESSION_ID, SESSION_ID,
                                                  TOKEN_ID)
from twinkle.server.telemetry.tracing import traced_operation
from twinkle.utils.logger import get_logger
from .backend import StateBackend
from .backend.factory import create_backend
from .cleanup_coordinator import ResourceCleanupCoordinator
from .config_manager import ConfigManager
from .count_publisher import ResourceCountPublisher
from .future_manager import FutureManager
from .model_manager import ModelManager
from .models import FutureFailureRecord, ModelRecord, SamplingSessionRecord, SessionRecord
from .sampling_manager import SamplingSessionManager
from .session_manager import SessionManager

logger = get_logger()

_PROCESSOR_QUOTA_PREFIX = 'processor_quota::'


def _clean_processor_reservations(existing: Any, *, now: float) -> dict[str, dict[str, Any]]:
    """Return only well-formed processor reservations whose leases are active."""
    if not isinstance(existing, dict):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for processor_id, reservation in existing.items():
        if not isinstance(processor_id, str) or not isinstance(reservation, dict):
            continue
        expires_at = reservation.get('lease_expires_at')
        if isinstance(expires_at, (int, float)) and float(expires_at) > now:
            active[processor_id] = dict(reservation)
    return active


def _reserve_processor_transform(
    existing: Any,
    *,
    processor_id: str,
    session_id: str,
    now: float,
    lease_seconds: float,
    limit: int,
) -> dict[str, dict[str, Any]]:
    reservations = _clean_processor_reservations(existing, now=now)
    if processor_id in reservations or len(reservations) < limit:
        reservations[processor_id] = {
            'session_id': session_id,
            'lease_expires_at': now + lease_seconds,
        }
    return reservations


def _renew_processor_transform(
    existing: Any,
    *,
    processor_id: str,
    now: float,
    lease_seconds: float,
) -> dict[str, dict[str, Any]]:
    reservations = _clean_processor_reservations(existing, now=now)
    reservation = reservations.get(processor_id)
    if reservation is not None:
        reservation['lease_expires_at'] = now + lease_seconds
    return reservations


def _release_processor_transform(
    existing: Any,
    *,
    processor_id: str,
    now: float,
) -> dict[str, dict[str, Any]]:
    reservations = _clean_processor_reservations(existing, now=now)
    reservations.pop(processor_id, None)
    return reservations


def _sweep_processor_transform(existing: Any, *, now: float) -> dict[str, dict[str, Any]]:
    return _clean_processor_reservations(existing, now=now)


class ServerState:
    """Unified server state management class.

    Composes five resource managers:

    - :class:`SessionManager` — client sessions
    - :class:`ModelManager` — registered models
    - :class:`SamplingSessionManager` — sampling sessions
    - :class:`FutureManager` — async task futures
    - :class:`ConfigManager` — key-value configuration

    Each Ray Serve worker owns one process-local instance, bound directly to a
    shared :class:`StateBackend`.

    Cleanup start-up: NOT from FastAPI lifespan startup. Ray Serve binds
    ``servable_object`` *after* lifespan startup, so ``start_cleanup_task()`` is
    lazy-started on the first request by
    ``deployment.LazyCleanupMixin._ensure_state_cleanup_started`` (the reason is
    spelled out in ``build_deployment_app``'s lifespan comment). Cleanup
    orchestration and cleanup-leader election live in
    :class:`ResourceCleanupCoordinator`; resource-count publishing lives in
    :class:`ResourceCountPublisher`. ``start_cleanup_task`` is idempotent, which is
    what makes per-request invocation safe.
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

        # Cleanup orchestration, leader election and resource-count publishing are
        # delegated. ``metrics_update_interval`` controls how often the leader pushes
        # counts into the MetricsRegistry cache. ``on_become_leader`` /
        # ``on_lose_leader`` wire leader identity to the publisher's lifecycle so only
        # the leader publishes, while keeping the coordinator itself free
        # of any telemetry dependency.
        _managers = {
            'sessions': self._session_mgr,
            'models': self._model_mgr,
            'sampling_sessions': self._sampling_mgr,
            'futures': self._future_mgr,
        }
        self._count_publisher = ResourceCountPublisher(
            [
                ('active_sessions', self._session_mgr),
                ('active_models', self._model_mgr),
                ('active_sampling_sessions', self._sampling_mgr),
                ('active_futures', self._future_mgr),
            ],
            interval=float(metrics_update_interval),
        )
        self._cleanup = ResourceCleanupCoordinator(
            self._backend,
            _managers,
            cleanup_interval=cleanup_interval,
            expiration_timeout=expiration_timeout,
            sweep_processor_quotas=self.sweep_processor_quotas,
            on_become_leader=self._count_publisher.start,
            on_lose_leader=self._on_lose_leader,
        )

    async def _on_lose_leader(self) -> None:
        await self._count_publisher.stop()
        self._count_publisher.clear()

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

    # ----- Processor Quota Management -----

    @staticmethod
    def _processor_quota_key(token: str) -> str:
        return f'{_PROCESSOR_QUOTA_PREFIX}{token}'

    @staticmethod
    def _processor_quota_ttl(lease_seconds: float) -> int:
        # The key-level TTL is only stale-key hygiene. Individual entries carry
        # their own deadlines and are cleaned atomically on every operation.
        return max(1, math.ceil(lease_seconds * 2))

    async def reserve_processor_quota(
        self,
        token: str,
        processor_id: str,
        session_id: str,
        *,
        limit: int,
        lease_seconds: float,
    ) -> None:
        """Atomically reserve one cluster-wide processor slot for ``token``."""
        now = time.time()
        reservations = await self._backend.update_atomic(
            self._processor_quota_key(token),
            functools.partial(
                _reserve_processor_transform,
                processor_id=processor_id,
                session_id=session_id,
                now=now,
                lease_seconds=lease_seconds,
                limit=limit,
            ),
            ttl=self._processor_quota_ttl(lease_seconds),
        )
        if not isinstance(reservations, dict) or processor_id not in reservations:
            raise ResourceQuotaExceededError(f'Per-user processor quota ({limit}) reached for token {token[:8]}...')

    async def renew_processor_quota(
        self,
        token: str,
        processor_id: str,
        *,
        lease_seconds: float,
    ) -> bool:
        """Renew an existing reservation; never recreate an expired lease."""
        now = time.time()
        reservations = await self._backend.update_atomic(
            self._processor_quota_key(token),
            functools.partial(
                _renew_processor_transform,
                processor_id=processor_id,
                now=now,
                lease_seconds=lease_seconds,
            ),
            ttl=self._processor_quota_ttl(lease_seconds),
        )
        return isinstance(reservations, dict) and processor_id in reservations

    async def release_processor_quota(self, token: str, processor_id: str, *, lease_seconds: float = 30.0) -> None:
        """Idempotently release a processor reservation."""
        await self._backend.update_atomic(
            self._processor_quota_key(token),
            functools.partial(_release_processor_transform, processor_id=processor_id, now=time.time()),
            ttl=self._processor_quota_ttl(lease_seconds),
        )

    async def sweep_processor_quotas(self, *, lease_seconds: float = 30.0) -> None:
        """Remove expired leases from every persisted processor quota map."""
        now = time.time()
        keys = await self._backend.keys(f'{_PROCESSOR_QUOTA_PREFIX}*')
        for key in keys:
            await self._backend.update_atomic(
                key,
                functools.partial(_sweep_processor_transform, now=now),
                ttl=self._processor_quota_ttl(lease_seconds),
            )

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

    async def claim_seq(self, dedup_key: str, request_id: str, ttl: int) -> str | None:
        """Idempotency claim for a client seq_id.

        Atomically records ``dedup_key -> request_id`` if unseen and returns ``None``
        (caller proceeds to enqueue). If the key already exists, returns the prior
        ``request_id`` so the caller can return that task's envelope instead of
        enqueuing a duplicate. ``ttl`` bounds the dedup window.
        """
        if await self._backend.set_nx(dedup_key, request_id, ttl=ttl):
            return None
        return await self._backend.get(dedup_key)

    async def release_seq(self, dedup_key: str) -> None:
        """Drop a seq dedup claim (used when the claimed request never enqueued, e.g.
        preflight rejected it) so a retry can be admitted rather than see a phantom."""
        await self._backend.delete(dedup_key)

    async def cancel_future(self, request_id: str) -> dict[str, Any]:
        """Best-effort cancel: drop the task iff it has not started running.

        Returns ``{'cancelled': bool, 'state': str}`` where ``state`` is the task's
        status after the attempt (``cancelled`` if just dropped or already cancelled,
        ``running``/``completed``/``failed`` if too late, ``not_found`` if unknown).
        """
        status = await self._future_mgr.cancel_if_pending(request_id)
        if status is None:
            return {'cancelled': False, 'state': 'not_found'}
        return {'cancelled': status == 'cancelled', 'state': status}

    async def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: str | None,
        reason: str | None = None,
        result: Any = None,
        failure: FutureFailureRecord | None = None,
        queue_state: str | None = None,
        queue_state_reason: str | None = None,
        replica_id: str | None = None,
        absolute_deadline: float | None = None,
    ) -> None:
        """Store task status with either a success result or domain failure.

        Supports the full task lifecycle:
        - PENDING: Task created, waiting to be processed
        - QUEUED: Task in queue waiting for execution
        - RUNNING: Task currently executing
        - COMPLETED: Task completed successfully (result required)
        - FAILED: Task failed (failure contains protocol-independent details)

        Args:
            request_id: Unique identifier for the request.
            status: Task status string (pending/queued/running/completed/failed).
            model_id: Optional associated model_id.
            reason: Optional reason string.
            result: Optional success result data.
            failure: Optional protocol-independent failure record.
            queue_state: Optional queue state for tinker client (active/paused_rate_limit/paused_capacity).
            queue_state_reason: Optional reason for the queue state.
        """
        await self._future_mgr.store_status(
            request_id=request_id,
            status=status,
            model_id=model_id,
            reason=reason,
            result=result,
            failure=failure,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
            replica_id=replica_id,
            absolute_deadline=absolute_deadline,
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

        Delegates to :class:`ResourceCleanupCoordinator`.
        """
        return await self._cleanup.cleanup_expired_resources()

    async def touch_replica_last_seen(self, replica_id: str) -> None:
        """Refresh a replica's liveness timestamp in the shared registry."""
        await self._model_mgr.touch_replica_last_seen(replica_id)

    # ----- Cleanup + leader election (delegated to ResourceCleanupCoordinator) -----

    @property
    def _is_leader(self) -> bool:
        return self._cleanup._is_leader

    @property
    def _leader_id(self) -> str:
        return self._cleanup._leader_id

    async def _try_acquire_or_renew(self) -> None:
        await self._cleanup._try_acquire_or_renew()

    async def start_cleanup_task(self) -> bool:
        """Start the background cleanup + leader-election tasks.

        Returns True if tasks were started, False if already running. The
        idempotency guard lives inside the coordinator's ``start`` so a
        per-request lazy invocation cannot double-start.
        """
        return await self._cleanup.start()

    async def stop_cleanup_task(self) -> bool:
        """Stop the background cleanup + leader-election tasks.

        Returns True if tasks were stopped, False if not running.
        """
        return await self._cleanup.stop()

    async def get_cleanup_stats(self) -> dict[str, Any]:
        """Get current cleanup configuration and resource counts."""
        return await self._cleanup.get_cleanup_stats()


# ---------------------------------------------------------------------------
# Per-process ServerState cache
# ---------------------------------------------------------------------------
#
# Each Ray Serve worker binds one ``ServerState`` instance to the shared
# ``StateBackend`` for the lifetime of the process — the cleanup loop and
# leader-election loop are started exactly once per worker (see
# ``start_cleanup_task``). Callers use ``cache_key`` purely for per-process
# deduplication; cross-worker coordination happens inside the shared backend,
# not in this dict.

_PROCESS_STATE_CACHE: dict[str, ServerState] = {}

# ServerState policy defaults. Used when neither an explicit argument nor a
# launcher-propagated env var (``ServerStateArgs.from_env``) supplies a value.
_DEFAULT_EXPIRATION_TIMEOUT = 86400.0  # 24 hours in seconds
_DEFAULT_CLEANUP_INTERVAL = 3600.0  # 1 hour in seconds
_DEFAULT_PER_TOKEN_MODEL_LIMIT = 30
_DEFAULT_METRICS_UPDATE_INTERVAL = 15.0


def get_server_state(cache_key: str = 'twinkle_server_state',
                     backend: StateBackend | None = None,
                     persistence_config: PersistenceConfig | None = None,
                     expiration_timeout: float | None = None,
                     cleanup_interval: float | None = None,
                     per_token_model_limit: int | None = None,
                     metrics_update_interval: float | None = None) -> ServerState:
    """Return a process-local :class:`ServerState` bound directly to the backend.

    Within one process the same ``cache_key`` returns the same cached instance
    so repeated callers share one ``ServerState`` and the cleanup loop is
    started exactly once. Cross-worker consistency comes from the shared
    :class:`StateBackend` rather than from any singleton in this process.

    Args:
        cache_key: Cache key for the per-process ``ServerState`` instance.
            (Formerly ``actor_name``; the parameter never carried actor
            semantics.)
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

    cached = _PROCESS_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Resolve the ServerState policy: an explicit argument wins, else the
    # launcher-propagated env (so a non-gateway worker honours the operator's
    # YAML instead of the hardcoded default), else the module default.
    from twinkle.server.config.application_spec import ServerStateArgs
    env_policy = ServerStateArgs.from_env()

    def _resolve(explicit, env_value, default):
        if explicit is not None:
            return explicit
        if env_value is not None:
            return env_value
        return default

    expiration_timeout = _resolve(expiration_timeout, getattr(env_policy, 'expiration_timeout', None),
                                  _DEFAULT_EXPIRATION_TIMEOUT)
    cleanup_interval = _resolve(cleanup_interval, getattr(env_policy, 'cleanup_interval', None),
                                _DEFAULT_CLEANUP_INTERVAL)
    per_token_model_limit = _resolve(per_token_model_limit, getattr(env_policy, 'per_token_model_limit', None),
                                     _DEFAULT_PER_TOKEN_MODEL_LIMIT)
    metrics_update_interval = _resolve(metrics_update_interval, getattr(env_policy, 'metrics_update_interval', None),
                                       _DEFAULT_METRICS_UPDATE_INTERVAL)

    state = ServerState(
        backend=backend,
        persistence_config=persistence_config,
        expiration_timeout=expiration_timeout,
        cleanup_interval=cleanup_interval,
        per_token_model_limit=per_token_model_limit,
        metrics_update_interval=metrics_update_interval,
    )
    _PROCESS_STATE_CACHE[cache_key] = state
    logger.info(
        'ServerState policy in effect: per_token_model_limit=%s expiration_timeout=%s '
        'cleanup_interval=%s metrics_update_interval=%s (resolution: explicit>env>default)', per_token_model_limit,
        expiration_timeout, cleanup_interval, metrics_update_interval)
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
