# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from twinkle.utils.logger import get_logger
import ray

logger = get_logger()

class ServerState:
    """
    Unified server state management class.
    
    This class combines the functionality of:
    1. Session management (create, touch, heartbeat)
    2. Model registration and tracking
    3. Sampling session management
    4. Async future storage and retrieval
    5. Configuration storage
    
    All methods are designed to be used with Ray actors for distributed state.
    """

    def __init__(self, 
                 expiration_timeout: float = 86400.0,  # 24 hours in seconds
                 cleanup_interval: float = 3600.0) -> None:  # 1 hour in seconds
        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # Model registration
        self.models: Dict[str, Dict[str, Any]] = {}
        # Sampling session tracking
        self.sampling_sessions: Dict[str, Dict[str, Any]] = {}
        # Async future results
        self.futures: Dict[str, Dict[str, Any]] = {}
        # Configuration storage
        self.config: Dict[str, Any] = {}
        
        # Cleanup configuration
        self.expiration_timeout = expiration_timeout
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False

    # ----- Session Management -----

    def create_session(self, payload: Dict[str, Any]) -> str:
        """
        Create a new session with the given payload.
        
        Args:
            payload: Session configuration containing optional session_id, tags, etc.
            
        Returns:
            The session_id for the created session
        """
        session_id = payload.get('session_id') or f"session_{uuid.uuid4().hex}"
        self.sessions[session_id] = {
            'tags': list(payload.get('tags') or []),
            'user_metadata': payload.get('user_metadata') or {},
            'sdk_version': payload.get('sdk_version'),
            'created_at': datetime.now().isoformat(),
        }
        return session_id

    def touch_session(self, session_id: str) -> bool:
        """
        Update session heartbeat timestamp.
        
        Args:
            session_id: The session to touch
            
        Returns:
            True if session exists and was touched, False otherwise
        """
        if session_id not in self.sessions:
            return False
        self.sessions[session_id]['last_heartbeat'] = time.time()
        return True

    # ----- Model Registration -----

    def register_model(self,
                       payload: Dict[str, Any],
                       model_id: Optional[str] = None,
                       token: Optional[str] = None) -> str:
        """
        Register a new model with the server state.
        
        Args:
            payload: Model configuration containing base_model, lora_config, etc.
            model_id: Optional explicit model_id, otherwise auto-generated
            token: Optional user token for tracking ownership
            
        Returns:
            The model_id for the registered model
        """
        _time = datetime.now().strftime('%Y%m%d_%H%M%S')
        _model_id: str = model_id or payload.get(
            'model_id'
        ) or f"{_time}-{payload.get('base_model', 'model')}-{uuid.uuid4().hex[:8]}"
        _model_id = re.sub(r'[^\w\-]', '_', _model_id)

        self.models[_model_id] = {
            'session_id': payload.get('session_id'),
            'model_seq_id': payload.get('model_seq_id'),
            'base_model': payload.get('base_model'),
            'user_metadata': payload.get('user_metadata') or {},
            'lora_config': payload.get('lora_config'),
            'token': token,  # Store token for adapter cleanup integration
            'created_at': datetime.now().isoformat(),
        }
        return _model_id

    def unload_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: The model to unload
            
        Returns:
            True if model was found and removed, False otherwise
        """
        return self.models.pop(model_id, None) is not None

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a registered model."""
        return self.models.get(model_id)

    # ----- Sampling Session Management -----

    def create_sampling_session(
            self,
            payload: Dict[str, Any],
            sampling_session_id: Optional[str] = None) -> str:
        """
        Create a new sampling session.
        
        Args:
            payload: Session configuration
            sampling_session_id: Optional explicit ID
            
        Returns:
            The sampling_session_id
        """
        _sampling_session_id: str = sampling_session_id or payload.get(
            'sampling_session_id') or f"sampling_{uuid.uuid4().hex}"
        self.sampling_sessions[_sampling_session_id] = {
            'session_id': payload.get('session_id'),
            'seq_id': payload.get('sampling_session_seq_id'),
            'base_model': payload.get('base_model'),
            'model_path': payload.get('model_path'),
            'created_at': datetime.now().isoformat(),
        }
        return _sampling_session_id

    def get_sampling_session(self, sampling_session_id: str) -> Optional[Dict[str, Any]]:
        """Get a sampling session by ID."""
        return self.sampling_sessions.get(sampling_session_id)

    # ----- Future Management -----

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored future result."""
        return self.futures.get(request_id)

    def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: Optional[str],
        reason: Optional[str] = None,
        result: Any = None,
        queue_state: Optional[str] = None,
        queue_state_reason: Optional[str] = None,
    ) -> None:
        """Store task status with optional result.
        
        This method supports the full task lifecycle:
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
        # Serialize result if it has model_dump method
        if result is not None and hasattr(result, 'model_dump'):
            result = result.model_dump()
        
        future_data: Dict[str, Any] = {
            'status': status,
            'model_id': model_id,
            'updated_at': datetime.now().isoformat(),
        }
        
        # Include reason for rate_limited status
        if reason is not None:
            future_data['reason'] = reason
        
        # Include result for completed/failed status
        if result is not None:
            future_data['result'] = result
        
        # Include queue_state and queue_state_reason for tinker client compatibility
        if queue_state is not None:
            future_data['queue_state'] = queue_state
        if queue_state_reason is not None:
            future_data['queue_state_reason'] = queue_state_reason
        
        # Update or create the future entry
        if request_id in self.futures:
            self.futures[request_id].update(future_data)
        else:
            future_data['created_at'] = datetime.now().isoformat()
            self.futures[request_id] = future_data

    # ----- Config Management (from ConfigRegistry) -----

    def add_config(self, key: str, value: Any):
        """
        Add or update a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def add_or_get(self, key: str, value: Any) -> Any:
        """
        Add a config if not exists, otherwise return existing value.
        
        Args:
            key: Configuration key
            value: Value to add if key doesn't exist
            
        Returns:
            The existing or newly added value
        """
        if key in self.config:
            return self.config[key]
        self.config[key] = value
        return value

    def get_config(self, key: str) -> Optional[Any]:
        """Get a configuration value by key."""
        return self.config.get(key)

    def pop_config(self, key: str) -> Optional[Any]:
        """Remove and return a configuration value."""
        return self.config.pop(key, None)

    def clear_config(self):
        """Clear all configuration values."""
        self.config.clear()

    # ----- Resource Cleanup -----

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse ISO format timestamp to unix timestamp.
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            Unix timestamp (seconds since epoch)
        """
        try:
            dt = datetime.fromisoformat(timestamp_str)
            return dt.timestamp()
        except (ValueError, AttributeError):
            # If parsing fails, return current time to avoid keeping invalid entries
            return time.time()

    def cleanup_expired_resources(self) -> Dict[str, int]:
        """Clean up expired sessions, models, sampling_sessions, and futures.
        
        Resources are considered expired if they haven't been accessed for longer
        than the expiration_timeout period. For sessions, we check last_heartbeat
        (or created_at if no heartbeat exists). For other resources, we check created_at.
        
        Returns:
            Dict with counts of cleaned up resources by type
        """
        current_time = time.time()
        cutoff_time = current_time - self.expiration_timeout
        
        cleanup_stats = {
            'sessions': 0,
            'models': 0,
            'sampling_sessions': 0,
            'futures': 0,
        }
        
        # Clean up expired sessions
        expired_session_ids = []
        for session_id, session_data in self.sessions.items():
            # Use last_heartbeat if available, otherwise created_at
            last_activity = session_data.get('last_heartbeat')
            if last_activity is None:
                created_at_str = session_data.get('created_at')
                if created_at_str:
                    last_activity = self._parse_timestamp(created_at_str)
                else:
                    last_activity = 0
            
            if last_activity < cutoff_time:
                expired_session_ids.append(session_id)
        
        for session_id in expired_session_ids:
            del self.sessions[session_id]
            cleanup_stats['sessions'] += 1
        
        # Clean up expired models (check by session_id association or created_at)
        expired_model_ids = []
        for model_id, model_data in self.models.items():
            # First check if the model's session has been cleaned up
            session_id = model_data.get('session_id')
            if session_id and session_id in expired_session_ids:
                expired_model_ids.append(model_id)
            else:
                # Check if model itself is expired by created_at
                created_at_str = model_data.get('created_at')
                if created_at_str:
                    created_at = self._parse_timestamp(created_at_str)
                    if created_at < cutoff_time:
                        expired_model_ids.append(model_id)
        
        for model_id in expired_model_ids:
            del self.models[model_id]
            cleanup_stats['models'] += 1
        
        # Clean up expired sampling sessions
        expired_sampling_ids = []
        for sampling_id, sampling_data in self.sampling_sessions.items():
            # Check by session_id association or created_at
            session_id = sampling_data.get('session_id')
            if session_id and session_id in expired_session_ids:
                expired_sampling_ids.append(sampling_id)
            else:
                created_at_str = sampling_data.get('created_at')
                if created_at_str:
                    created_at = self._parse_timestamp(created_at_str)
                    if created_at < cutoff_time:
                        expired_sampling_ids.append(sampling_id)
        
        for sampling_id in expired_sampling_ids:
            del self.sampling_sessions[sampling_id]
            cleanup_stats['sampling_sessions'] += 1
        
        # Clean up expired futures (use created_at or updated_at)
        expired_future_ids = []
        for request_id, future_data in self.futures.items():
            # Use updated_at if available, otherwise created_at
            timestamp_str = future_data.get('updated_at') or future_data.get('created_at')
            if timestamp_str:
                timestamp = self._parse_timestamp(timestamp_str)
                if timestamp < cutoff_time:
                    expired_future_ids.append(request_id)
        
        for request_id in expired_future_ids:
            del self.futures[request_id]
            cleanup_stats['futures'] += 1
        
        return cleanup_stats

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired resources.
        
        This task runs continuously and triggers cleanup at regular intervals
        defined by cleanup_interval.
        """
        while self._cleanup_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                stats = self.cleanup_expired_resources()
                # Log cleanup stats (in production, you might want to use proper logging)
                if any(stats.values()):
                    logger.debug(f"[ServerState Cleanup] Removed expired resources: {stats}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but don't crash the cleanup task
                logger.warning(f"[ServerState Cleanup] Error during cleanup: {e}")
                continue

    def start_cleanup_task(self) -> bool:
        """Start the background cleanup task.
        
        Returns:
            True if task was started, False if already running
        """
        if self._cleanup_running:
            return False
        
        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        return True

    def stop_cleanup_task(self) -> bool:
        """Stop the background cleanup task.
        
        Returns:
            True if task was stopped, False if not running
        """
        if not self._cleanup_running:
            return False
        
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        return True

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get current cleanup configuration and status.
        
        Returns:
            Dict with cleanup configuration and task status
        """
        return {
            'expiration_timeout': self.expiration_timeout,
            'cleanup_interval': self.cleanup_interval,
            'cleanup_running': self._cleanup_running,
            'resource_counts': {
                'sessions': len(self.sessions),
                'models': len(self.models),
                'sampling_sessions': len(self.sampling_sessions),
                'futures': len(self.futures),
            }
        }


class ServerStateProxy:
    """
    Proxy for interacting with ServerState Ray actor.
    
    This class wraps Ray remote calls to provide a synchronous-looking API
    for interacting with the distributed ServerState actor.
    """

    def __init__(self, actor_handle):
        self._actor = actor_handle

    # ----- Session Management -----

    def create_session(self, payload: Dict[str, Any]) -> str:
        return ray.get(self._actor.create_session.remote(payload))

    def touch_session(self, session_id: str) -> bool:
        return ray.get(self._actor.touch_session.remote(session_id))

    # ----- Model Registration -----

    def register_model(self,
                       payload: Dict[str, Any],
                       model_id: Optional[str] = None,
                       token: Optional[str] = None) -> str:
        return ray.get(self._actor.register_model.remote(payload, model_id, token))

    def unload_model(self, model_id: str) -> bool:
        return ray.get(self._actor.unload_model.remote(model_id))

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.get_model_metadata.remote(model_id))

    # ----- Sampling Session Management -----

    def create_sampling_session(
            self,
            payload: Dict[str, Any],
            sampling_session_id: Optional[str] = None) -> str:
        return ray.get(
            self._actor.create_sampling_session.remote(payload, sampling_session_id))

    def get_sampling_session(self, sampling_session_id: str) -> Optional[Dict[str, Any]]:
        """Get a sampling session by ID."""
        return ray.get(self._actor.get_sampling_session.remote(sampling_session_id))

    # ----- Future Management -----

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.get_future.remote(request_id))

    def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: Optional[str],
        reason: Optional[str] = None,
        result: Any = None,
        queue_state: Optional[str] = None,
        queue_state_reason: Optional[str] = None,
    ) -> None:
        """Store task status with optional result (synchronous)."""
        ray.get(self._actor.store_future_status.remote(
            request_id, status, model_id, reason, result, queue_state, queue_state_reason
        ))

    # ----- Config Management -----

    def add_config(self, key: str, value: Any):
        return ray.get(self._actor.add_config.remote(key, value))

    def add_or_get(self, key: str, value: Any) -> Any:
        return ray.get(self._actor.add_or_get.remote(key, value))

    def get_config(self, key: str) -> Optional[Any]:
        return ray.get(self._actor.get_config.remote(key))

    def pop_config(self, key: str) -> Optional[Any]:
        return ray.get(self._actor.pop_config.remote(key))

    def clear_config(self):
        return ray.get(self._actor.clear_config.remote())

    # ----- Resource Cleanup -----

    def cleanup_expired_resources(self) -> Dict[str, int]:
        """Manually trigger cleanup of expired resources.
        
        Returns:
            Dict with counts of cleaned up resources by type
        """
        return ray.get(self._actor.cleanup_expired_resources.remote())

    def start_cleanup_task(self) -> bool:
        """Start the background cleanup task.
        
        Returns:
            True if task was started, False if already running
        """
        return ray.get(self._actor.start_cleanup_task.remote())

    def stop_cleanup_task(self) -> bool:
        """Stop the background cleanup task.
        
        Returns:
            True if task was stopped, False if not running
        """
        return ray.get(self._actor.stop_cleanup_task.remote())

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get current cleanup configuration and status.
        
        Returns:
            Dict with cleanup configuration and task status
        """
        return ray.get(self._actor.get_cleanup_stats.remote())


def get_server_state(actor_name: str = 'twinkle_server_state',
                     auto_start_cleanup: bool = True) -> ServerStateProxy:
    """
    Get or create the ServerState Ray actor.
    
    This function ensures only one ServerState actor exists with the given name.
    It uses a detached actor so the state persists across driver restarts.
    
    Args:
        actor_name: Name for the Ray actor (default: 'twinkle_server_state')
        auto_start_cleanup: Whether to automatically start the cleanup task (default: True)
        
    Returns:
        A ServerStateProxy for interacting with the actor
    """
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        try:
            _ServerState = ray.remote(ServerState)
            actor = _ServerState.options(name=actor_name, lifetime='detached').remote()
            # Start cleanup task for newly created actor
            if auto_start_cleanup:
                try:
                    ray.get(actor.start_cleanup_task.remote())
                except Exception as e:
                    logger.debug(f"[ServerState] Warning: Failed to start cleanup task: {e}")
        except ValueError:
            actor = ray.get_actor(actor_name)
    assert actor is not None
    return ServerStateProxy(actor)

