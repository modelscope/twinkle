# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import ray
import re
import time
import uuid
from datetime import datetime
from typing import Any

from twinkle.utils.logger import get_logger
from .config_manager import ConfigManager
from .future_manager import FutureManager
from .model_manager import ModelManager
from .models import ModelRecord, SamplingSessionRecord, SessionRecord
from .sampling_manager import SamplingSessionManager
from .session_manager import SessionManager

logger = get_logger()


class ServerState:
    """
    Unified server state management class.

    Composes five resource managers:
    - SessionManager       — client sessions
    - ModelManager         — registered models
    - SamplingSessionManager — sampling sessions
    - FutureManager        — async task futures
    - ConfigManager        — key-value configuration

    All methods are designed to be used with Ray actors for distributed state.
    """

    def __init__(
            self,
            expiration_timeout: float = 86400.0,  # 24 hours in seconds
            cleanup_interval: float = 3600.0,  # 1 hour in seconds
            per_token_model_limit: int = 30,
            **kwargs) -> None:
        self._session_mgr = SessionManager(expiration_timeout)
        self._model_mgr = ModelManager(expiration_timeout, per_token_model_limit)
        self._sampling_mgr = SamplingSessionManager(expiration_timeout)
        self._future_mgr = FutureManager(expiration_timeout)
        self._config_mgr = ConfigManager()

        self.expiration_timeout = expiration_timeout
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_running = False

    # ----- Session Management -----

    def create_session(self, payload: dict[str, Any]) -> str:
        """Create a new session with the given payload.

        Args:
            payload: Session configuration containing optional session_id, tags, etc.

        Returns:
            The session_id for the created session.
        """
        session_id = payload.get('session_id') or f'session_{uuid.uuid4().hex}'
        record = SessionRecord(
            tags=list(payload.get('tags') or []),
            user_metadata=payload.get('user_metadata') or {},
            sdk_version=payload.get('sdk_version'),
        )
        self._session_mgr.add(session_id, record)
        return session_id

    def touch_session(self, session_id: str) -> bool:
        """Update session heartbeat timestamp.

        Returns:
            True if the session exists and was touched, False otherwise.
        """
        return self._session_mgr.touch(session_id)

    def get_session_last_heartbeat(self, session_id: str) -> float | None:
        """Get the last heartbeat timestamp for a session.

        Returns:
            Last heartbeat timestamp, or None if the session does not exist.
        """
        return self._session_mgr.get_last_heartbeat(session_id)

    # ----- Model Registration -----

    def register_model(self, payload: dict[str, Any], token: str, model_id: str | None = None) -> str:
        """Register a new model with the server state.

        Args:
            payload: Model configuration containing base_model, lora_config, etc.
            token: User token that owns this model. Required.
            model_id: Optional explicit model_id; otherwise auto-generated.

        Returns:
            The model_id for the registered model.
        """
        _time = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_id: str = model_id or payload.get(
            'model_id') or f"{_time}-{payload.get('base_model', 'model')}-{uuid.uuid4().hex[:8]}"
        _model_id = re.sub(r'[^\w\-]', '_', _model_id)

        record = ModelRecord(
            session_id=payload.get('session_id'),
            model_seq_id=payload.get('model_seq_id'),
            base_model=payload.get('base_model'),
            user_metadata=payload.get('user_metadata') or {},
            lora_config=payload.get('lora_config'),
            token=token,
        )
        self._model_mgr.add(_model_id, record)
        return _model_id

    def unload_model(self, model_id: str) -> bool:
        """Remove a model from the registry.

        Returns:
            True if the model was found and removed, False otherwise.
        """
        return self._model_mgr.remove(model_id)

    def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
        """Get metadata for a registered model as a plain dict."""
        record = self._model_mgr.get(model_id)
        return record.model_dump() if record is not None else None

    # ----- Sampling Session Management -----

    def create_sampling_session(self, payload: dict[str, Any], sampling_session_id: str | None = None) -> str:
        """Create a new sampling session.

        Args:
            payload: Session configuration.
            sampling_session_id: Optional explicit ID.

        Returns:
            The sampling_session_id.
        """
        _sampling_session_id: str = sampling_session_id or payload.get(
            'sampling_session_id') or f'sampling_{uuid.uuid4().hex}'
        record = SamplingSessionRecord(
            session_id=payload.get('session_id'),
            seq_id=payload.get('sampling_session_seq_id'),
            base_model=payload.get('base_model'),
            model_path=payload.get('model_path'),
        )
        self._sampling_mgr.add(_sampling_session_id, record)
        return _sampling_session_id

    def get_sampling_session(self, sampling_session_id: str) -> dict[str, Any] | None:
        """Get a sampling session by ID as a plain dict."""
        record = self._sampling_mgr.get(sampling_session_id)
        return record.model_dump() if record is not None else None

    # ----- Future Management -----

    def get_future(self, request_id: str) -> dict[str, Any] | None:
        """Retrieve a stored future result as a plain dict."""
        record = self._future_mgr.get(request_id)
        return record.model_dump() if record is not None else None

    def store_future_status(
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
        self._future_mgr.store_status(
            request_id=request_id,
            status=status,
            model_id=model_id,
            reason=reason,
            result=result,
            queue_state=queue_state,
            queue_state_reason=queue_state_reason,
        )

    # ----- Config Management -----

    def add_config(self, key: str, value: Any) -> None:
        """Add or update a configuration value."""
        self._config_mgr.add(key, value)

    def add_or_get(self, key: str, value: Any) -> Any:
        """Add a config value if the key does not exist; otherwise return the existing value."""
        return self._config_mgr.add_or_get(key, value)

    def get_config(self, key: str) -> Any | None:
        """Get a configuration value by key."""
        return self._config_mgr.get(key)

    def pop_config(self, key: str) -> Any | None:
        """Remove and return a configuration value."""
        return self._config_mgr.pop(key)

    def clear_config(self) -> None:
        """Clear all configuration values."""
        self._config_mgr.clear()

    # ----- Resource Cleanup -----

    def cleanup_expired_resources(self) -> dict[str, int]:
        """Clean up expired sessions, models, sampling_sessions, and futures.

        Sessions expire based on last_heartbeat (or created_at).  Models and
        sampling sessions are also cascade-expired when their owning session
        expires.  Futures expire based on updated_at (or created_at).

        Returns:
            Dict with counts of cleaned up resources by type.
        """
        current_time = time.time()
        cutoff_time = current_time - self.expiration_timeout

        # Collect expired session IDs first for cascade logic
        expired_session_ids = self._session_mgr.get_expired_ids(cutoff_time)

        # Perform actual cleanup in dependency order
        sessions_removed = self._session_mgr.cleanup_expired(cutoff_time)
        models_removed = self._model_mgr.cleanup_expired(cutoff_time, expired_session_ids)
        samplings_removed = self._sampling_mgr.cleanup_expired(cutoff_time, expired_session_ids)
        futures_removed = self._future_mgr.cleanup_expired(cutoff_time)

        return {
            'sessions': sessions_removed,
            'models': models_removed,
            'sampling_sessions': samplings_removed,
            'futures': futures_removed,
        }

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired resources."""
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                stats = self.cleanup_expired_resources()
                if any(stats.values()):
                    logger.debug(f'[ServerState Cleanup] Removed expired resources: {stats}')
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f'[ServerState Cleanup] Error during cleanup: {e}')
                continue

    def start_cleanup_task(self) -> bool:
        """Start the background cleanup task.

        Returns:
            True if task was started, False if already running.
        """
        if self._cleanup_running:
            return False
        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        return True

    def stop_cleanup_task(self) -> bool:
        """Stop the background cleanup task.

        Returns:
            True if task was stopped, False if not running.
        """
        if not self._cleanup_running:
            return False
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        return True

    def get_cleanup_stats(self) -> dict[str, Any]:
        """Get current cleanup configuration and resource counts.

        Returns:
            Dict with cleanup configuration and task status.
        """
        return {
            'expiration_timeout': self.expiration_timeout,
            'cleanup_interval': self.cleanup_interval,
            'cleanup_running': self._cleanup_running,
            'resource_counts': {
                'sessions': self._session_mgr.count(),
                'models': self._model_mgr.count(),
                'sampling_sessions': self._sampling_mgr.count(),
                'futures': self._future_mgr.count(),
            },
        }


# ---------------------------------------------------------------------------
# Ray proxy
# ---------------------------------------------------------------------------


class ServerStateProxy:
    """
    Proxy for interacting with a ServerState Ray actor.

    Wraps Ray remote calls to provide a synchronous-looking API for
    interacting with the distributed ServerState actor.
    """

    def __init__(self, actor_handle) -> None:
        self._actor = actor_handle

    # ----- Session Management -----

    def create_session(self, payload: dict[str, Any]) -> str:
        return ray.get(self._actor.create_session.remote(payload))

    def touch_session(self, session_id: str) -> bool:
        return ray.get(self._actor.touch_session.remote(session_id))

    def get_session_last_heartbeat(self, session_id: str) -> float | None:
        return ray.get(self._actor.get_session_last_heartbeat.remote(session_id))

    # ----- Model Registration -----

    def register_model(self, payload: dict[str, Any], token: str, model_id: str | None = None) -> str:
        return ray.get(self._actor.register_model.remote(payload, token, model_id))

    def unload_model(self, model_id: str) -> bool:
        return ray.get(self._actor.unload_model.remote(model_id))

    def get_model_metadata(self, model_id: str) -> dict[str, Any] | None:
        return ray.get(self._actor.get_model_metadata.remote(model_id))

    # ----- Sampling Session Management -----

    def create_sampling_session(self, payload: dict[str, Any], sampling_session_id: str | None = None) -> str:
        return ray.get(self._actor.create_sampling_session.remote(payload, sampling_session_id))

    def get_sampling_session(self, sampling_session_id: str) -> dict[str, Any] | None:
        return ray.get(self._actor.get_sampling_session.remote(sampling_session_id))

    # ----- Future Management -----

    def get_future(self, request_id: str) -> dict[str, Any] | None:
        return ray.get(self._actor.get_future.remote(request_id))

    def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: str | None,
        reason: str | None = None,
        result: Any = None,
        queue_state: str | None = None,
        queue_state_reason: str | None = None,
    ) -> None:
        """Store task status with optional result (synchronous)."""
        ray.get(
            self._actor.store_future_status.remote(request_id, status, model_id, reason, result, queue_state,
                                                   queue_state_reason))

    # ----- Config Management -----

    def add_config(self, key: str, value: Any):
        return ray.get(self._actor.add_config.remote(key, value))

    def add_or_get(self, key: str, value: Any) -> Any:
        return ray.get(self._actor.add_or_get.remote(key, value))

    def get_config(self, key: str) -> Any | None:
        return ray.get(self._actor.get_config.remote(key))

    def pop_config(self, key: str) -> Any | None:
        return ray.get(self._actor.pop_config.remote(key))

    def clear_config(self):
        return ray.get(self._actor.clear_config.remote())

    # ----- Resource Cleanup -----

    def cleanup_expired_resources(self) -> dict[str, int]:
        return ray.get(self._actor.cleanup_expired_resources.remote())

    def start_cleanup_task(self) -> bool:
        return ray.get(self._actor.start_cleanup_task.remote())

    def stop_cleanup_task(self) -> bool:
        return ray.get(self._actor.stop_cleanup_task.remote())

    def get_cleanup_stats(self) -> dict[str, Any]:
        return ray.get(self._actor.get_cleanup_stats.remote())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_server_state(actor_name: str = 'twinkle_server_state', **kwargs) -> ServerStateProxy:
    """Get or create the ServerState Ray actor.

    Ensures only one ServerState actor exists with the given name.  Uses a
    detached actor so the state persists across driver restarts.

    Args:
        actor_name: Name for the Ray actor (default: 'twinkle_server_state').
        **kwargs: Additional keyword arguments passed to ServerState constructor
            (e.g., expiration_timeout, cleanup_interval).

    Returns:
        A ServerStateProxy for interacting with the actor.
    """
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        try:
            _ServerState = ray.remote(ServerState)
            actor = _ServerState.options(name=actor_name, lifetime='detached').remote(**kwargs)
            try:
                ray.get(actor.start_cleanup_task.remote())
            except Exception as e:
                logger.debug(f'[ServerState] Warning: Failed to start cleanup task: {e}')
        except ValueError:
            actor = ray.get_actor(actor_name)
    assert actor is not None
    return ServerStateProxy(actor)
