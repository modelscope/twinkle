# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import ray


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

    def __init__(self) -> None:
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
                       model_id: Optional[str] = None) -> str:
        """
        Register a new model with the server state.
        
        Args:
            payload: Model configuration containing base_model, lora_config, etc.
            model_id: Optional explicit model_id, otherwise auto-generated
            
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

    def store_future(self, request_id: str, result: Any,
                     model_id: Optional[str]):
        """
        Store the result of an async operation.
        
        Args:
            request_id: Unique identifier for the request
            result: The result to store
            model_id: Optional associated model_id
        """
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        self.futures[request_id] = {
            'status': 'completed',
            'result': result,
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
        }

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
                       model_id: Optional[str] = None) -> str:
        return ray.get(self._actor.register_model.remote(payload, model_id))

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

    async def store_future(self, request_id: str, result: Any,
                           model_id: Optional[str]):
        """Store future result asynchronously."""
        await self._actor.store_future.remote(request_id, result, model_id)

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.get_future.remote(request_id))

    def store_future_status(
        self,
        request_id: str,
        status: str,
        model_id: Optional[str],
        reason: Optional[str] = None,
        result: Any = None,
    ) -> None:
        """Store task status with optional result (synchronous)."""
        ray.get(self._actor.store_future_status.remote(
            request_id, status, model_id, reason, result
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


def get_server_state(actor_name: str = 'twinkle_server_state') -> ServerStateProxy:
    """
    Get or create the ServerState Ray actor.
    
    This function ensures only one ServerState actor exists with the given name.
    It uses a detached actor so the state persists across driver restarts.
    
    Args:
        actor_name: Name for the Ray actor (default: 'twinkle_server_state')
        
    Returns:
        A ServerStateProxy for interacting with the actor
    """
    _ServerState = ray.remote(ServerState)
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        try:
            actor = _ServerState.options(name=actor_name, lifetime='detached').remote()
        except ValueError:
            actor = ray.get_actor(actor_name)
    assert actor is not None
    return ServerStateProxy(actor)


# Alias for backward compatibility with ConfigRegistry usage
def init_config_registry(actor_name: str = 'twinkle_server_state') -> ServerStateProxy:
    """
    Initialize config registry (alias for get_server_state for backward compatibility).
    
    This function provides backward compatibility for code that was using
    init_config_registry from validation.py.
    """
    return get_server_state(actor_name)
