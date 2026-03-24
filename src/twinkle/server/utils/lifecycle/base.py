# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Base class for session-bound resource lifecycle management.

This module provides a generic mixin for managing resources (adapters, processors, etc.)
that are bound to user sessions and should expire when their session expires.
"""
from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy

from twinkle.utils.logger import get_logger

logger = get_logger()


class SessionResourceMixin:
    """Base mixin for managing session-bound resources with automatic expiration.

    This mixin tracks resources and automatically expires them when their
    associated session expires or when they exceed their maximum lifetime.

    Inheriting classes should:
    1. Call _init_resource_manager() in __init__
    2. Override _on_resource_expired() to handle resource-specific cleanup
    3. Optionally override _validate_registration() for custom validation

    Attributes:
        _resource_timeout: Session inactivity timeout in seconds.
        _resource_max_lifetime: Maximum lifetime in seconds for any resource.
        _resource_records: Dict mapping resource_id -> resource info dict.
    """

    # Type hint for state attribute that inheriting classes must provide
    state: ServerStateProxy

    # Resource type name for logging (override in subclass)
    _resource_type: str = 'resource'

    def _init_resource_manager(
        self,
        resource_timeout: float = 1800.0,
        resource_max_lifetime: float | None = None,
    ) -> None:
        """Initialize the resource manager.

        This should be called in the __init__ of the inheriting class.

        Args:
            resource_timeout: Timeout in seconds to determine if a session is alive.
                Default is 1800.0 (30 minutes).
            resource_max_lifetime: Maximum lifetime in seconds for a resource regardless
                of session liveness. None means no max lifetime limit.
        """
        self._resource_timeout = resource_timeout
        self._resource_max_lifetime = resource_max_lifetime

        # Resource lifecycle tracking
        # Dict mapping resource_id ->
        # {'token': str, 'session_id': str, 'created_at': float, 'state': dict, 'expiring': bool}
        self._resource_records: dict[str, dict[str, Any]] = {}

        # Countdown task
        self._resource_countdown_running = False
        self._countdown_task: asyncio.Task | None = None

    async def _is_session_alive(self, session_id: str) -> bool:
        """Check if a session is still alive via state proxy.

        Args:
            session_id: Session ID to check

        Returns:
            True if session is alive, False if expired or not found
        """
        if not session_id:
            return True  # No session association means always alive

        try:
            last_heartbeat = await self.state.get_session_last_heartbeat(session_id)
        except Exception as e:
            logger.warning(f'[{self._resource_type}Manager] Failed to check session liveness: {e}')
            return True  # Assume alive on error

        if last_heartbeat is None:
            return False  # Session doesn't exist

        # Check if session has timed out
        return (time.time() - last_heartbeat) < self._resource_timeout

    def _validate_registration(self, resource_id: str, token: str, session_id: str) -> None:
        """Validate before registering a resource. Override for custom validation.

        Args:
            resource_id: Resource identifier
            token: User token
            session_id: Session ID

        Raises:
            ValueError: If validation fails
            RuntimeError: If resource limit is reached
        """
        if not session_id:
            raise ValueError(f'session_id must be provided when registering {self._resource_type} {resource_id}')

    def _create_resource_record(self, token: str, session_id: str) -> dict[str, Any]:
        """Create a new resource record. Override to add custom fields.

        Args:
            token: User token
            session_id: Session ID

        Returns:
            Resource record dict
        """
        return {
            'token': token,
            'session_id': session_id,
            'created_at': time.time(),
            'state': {},
            'expiring': False,
        }

    def register_resource(self, resource_id: str, token: str, session_id: str) -> None:
        """Register a new resource for lifecycle tracking.

        Args:
            resource_id: Unique identifier of the resource.
            token: User token that owns this resource.
            session_id: Session ID to associate with this resource.

        Raises:
            ValueError: If session_id is None or empty.
            RuntimeError: If custom validation fails (e.g., limit reached).
        """
        self._validate_registration(resource_id, token, session_id)

        self._resource_records[resource_id] = self._create_resource_record(token, session_id)
        logger.debug(f'[{self._resource_type}Manager] Registered {self._resource_type} {resource_id} '
                     f'for token {token[:8]}... (session: {session_id})')

    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource from lifecycle tracking.

        Args:
            resource_id: Resource identifier to unregister.

        Returns:
            True if resource was found and removed, False otherwise.
        """
        if resource_id in self._resource_records:
            info = self._resource_records.pop(resource_id)
            token = info.get('token')
            logger.debug(f'[{self._resource_type}Manager] Unregistered {self._resource_type} {resource_id} '
                         f"for token {token[:8] if token else 'unknown'}...")
            return True
        return False

    def get_resource_info(self, resource_id: str) -> dict[str, Any] | None:
        """Get information about a registered resource.

        Args:
            resource_id: Resource identifier to query.

        Returns:
            Dict with resource information or None if not found.
        """
        return self._resource_records.get(resource_id)

    def set_resource_state(self, resource_id: str, key: str, value: Any) -> None:
        """Set a per-resource state value.

        This is intentionally generic so higher-level services can store
        resource-scoped state without maintaining separate side maps.
        """
        info = self._resource_records.get(resource_id)
        if info is None:
            return
        state = info.setdefault('state', {})
        state[key] = value

    def get_resource_state(self, resource_id: str, key: str, default: Any = None) -> Any:
        """Get a per-resource state value."""
        info = self._resource_records.get(resource_id)
        if info is None:
            return default
        state = info.get('state') or {}
        return state.get(key, default)

    def pop_resource_state(self, resource_id: str, key: str, default: Any = None) -> Any:
        """Pop a per-resource state value."""
        info = self._resource_records.get(resource_id)
        if info is None:
            return default
        state = info.get('state')
        if not isinstance(state, dict):
            return default
        return state.pop(key, default)

    def clear_resource_state(self, resource_id: str) -> None:
        """Clear all per-resource state values."""
        info = self._resource_records.get(resource_id)
        if info is None:
            return
        info['state'] = {}

    def assert_resource_exists(self, resource_id: str) -> None:
        """Validate that a resource exists and is not expiring.

        Raises:
            AssertionError: If resource not found or expiring.
        """
        info = self._resource_records.get(resource_id)
        assert resource_id and info is not None and not info.get('expiring'), \
            f'{self._resource_type} {resource_id} not found'

    @abstractmethod
    async def _on_resource_expired(self, resource_id: str) -> None:
        """Hook method called when a resource expires.

        This method must be implemented by inheriting classes to handle
        resource-specific expiration logic.

        Args:
            resource_id: Identifier of the expired resource.
        """
        raise NotImplementedError(f'_on_resource_expired must be implemented by {self.__class__.__name__}')

    async def _resource_countdown_loop(self) -> None:
        """Background task that monitors and handles expired resources.

        This task runs continuously and:
        1. Checks whether a resource has exceeded `_resource_max_lifetime` (if set)
        2. Checks session liveness for remaining resources
        3. Calls _on_resource_expired() for resources that have expired
        4. Removes expired resources from tracking
        """
        logger.debug(f'[{self._resource_type}Manager] Countdown task started '
                     f'(session_timeout={self._resource_timeout}s)')
        while self._resource_countdown_running:
            try:
                await asyncio.sleep(10)

                expired_resources: list[tuple[str, str | None, str | None]] = []
                # Create snapshot to avoid modification during iteration
                resource_snapshot = list(self._resource_records.items())
                for resource_id, info in resource_snapshot:
                    if info.get('expiring'):
                        continue

                    session_id = info.get('session_id')
                    created_at = info.get('created_at', 0.0)
                    now = time.time()

                    # Check max lifetime first (no async call needed)
                    if self._resource_max_lifetime and now - created_at >= self._resource_max_lifetime:
                        logger.debug(f'[{self._resource_type}Manager] {self._resource_type} {resource_id} '
                                     f'exceeded max lifetime ({self._resource_max_lifetime}s), marking as expired')
                        info['expiring'] = True
                        info['state'] = {}
                        token = info.get('token')
                        expired_resources.append((resource_id, token, session_id))
                        continue

                    try:
                        session_alive = await self._is_session_alive(session_id)
                    except Exception as e:
                        logger.warning(f'[{self._resource_type}Manager] Failed to check session liveness '
                                       f'for {resource_id}: {type(e).__name__}: {e}')
                        continue
                    session_expired = not session_alive
                    logger.debug(f'[{self._resource_type}Manager] {self._resource_type} {resource_id} session check '
                                 f'(session_id={session_id}, session_alive={not session_expired})')

                    if session_expired:
                        info['expiring'] = True
                        info['state'] = {}  # best-effort clear
                        token = info.get('token')
                        expired_resources.append((resource_id, token, session_id))

                for resource_id, _token, session_id in expired_resources:
                    success = False
                    try:
                        await self._on_resource_expired(resource_id)
                        logger.info(f'[{self._resource_type}Manager] {self._resource_type} {resource_id} expired '
                                    f'(reason=session_expired, session={session_id})')
                        success = True
                    except Exception as e:
                        logger.warning(f'[{self._resource_type}Manager] Error while expiring {self._resource_type} '
                                       f'{resource_id}: {e}')
                    finally:
                        if success:
                            self._resource_records.pop(resource_id, None)
                        else:
                            info = self._resource_records.get(resource_id)
                            if info is not None:
                                info['expiring'] = False

            except Exception as e:
                logger.warning(f'[{self._resource_type}Manager] Error in countdown loop: {e}')
                continue

        logger.debug(f'[{self._resource_type}Manager] Countdown task stopped')

    def _ensure_countdown_started(self) -> None:
        """Ensure the countdown task is started. Call from async context."""
        if not self._resource_countdown_running:
            self._resource_countdown_running = True
            self._countdown_task = asyncio.create_task(self._resource_countdown_loop())
            logger.debug(f'[{self._resource_type}Manager] Countdown task started')

    async def _async_ensure_countdown_started(self) -> None:
        """Async version for convenience."""
        self._ensure_countdown_started()

    def stop_resource_countdown(self) -> None:
        """Stop the background countdown task."""
        if self._resource_countdown_running:
            self._resource_countdown_running = False
            if self._countdown_task:
                self._countdown_task.cancel()
            logger.debug(f'[{self._resource_type}Manager] Countdown task stopped')
