# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Adapter Lifecycle Manager Mixin for Twinkle Server.

This module provides adapter lifecycle management as a mixin class that can be
inherited directly by services. It tracks adapter activity and provides interfaces
for registration, heartbeat updates, and expiration handling.

By inheriting this mixin, services can override the _on_adapter_expired() method
to handle expired adapters without using callbacks or polling.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy

from twinkle.utils.logger import get_logger

logger = get_logger()


class AdapterManagerMixin:
    """Mixin for adapter lifecycle management with session-based expiration.

    This mixin tracks adapter activity and automatically expires adapters
    when their associated session expires.

    Inheriting classes should:
    1. Call _init_adapter_manager() in __init__
    2. Override _on_adapter_expired() to customize expiration handling

    Attributes:
        _adapter_timeout: Session inactivity timeout in seconds used to determine if a session is alive.
        _adapter_max_lifetime: Maximum lifetime in seconds for any adapter, regardless of session liveness.
    """

    # Type hint for state attribute that inheriting classes must provide
    state: ServerStateProxy

    def _init_adapter_manager(
        self,
        adapter_timeout: float = 1800.0,
        adapter_max_lifetime: float = 36000.0,
    ) -> None:
        """Initialize the adapter manager.

        This should be called in the __init__ of the inheriting class.

        Args:
            adapter_timeout: Timeout in seconds used to check whether a session is still alive.
                Default is 1800.0 (30 minutes).
            adapter_max_lifetime: Maximum lifetime in seconds for an adapter regardless of session
                liveness. Adapters older than this are treated as expired. Default is 36000.0 (10 hours).
        """
        self._adapter_timeout = adapter_timeout
        self._adapter_max_lifetime = adapter_max_lifetime

        # Adapter lifecycle tracking
        # Dict mapping adapter_name ->
        # {'token': str, 'session_id': str, 'created_at': float, 'state': dict, 'expiring': bool}
        self._adapter_records: dict[str, dict[str, Any]] = {}

        # Countdown thread
        self._adapter_countdown_thread: threading.Thread | None = None
        self._adapter_countdown_running = False

    def register_adapter(self, adapter_name: str, token: str, session_id: str) -> None:
        """Register a new adapter for lifecycle tracking.

        The adapter will expire when its associated session expires.

        Args:
            adapter_name: Name of the adapter to register.
            token: User token that owns this adapter.
            session_id: Session ID to associate with this adapter. Must be a non-empty string.

        Raises:
            ValueError: If session_id is None or empty.
        """
        if not session_id:
            raise ValueError(f'session_id must be provided when registering adapter {adapter_name}')
        current_time = time.time()
        self._adapter_records[adapter_name] = {
            'token': token,
            'session_id': session_id,
            'created_at': current_time,
            'state': {},
            'expiring': False,
        }
        logger.debug(
            f'[AdapterManager] Registered adapter {adapter_name} for token {token[:8]}... (session: {session_id})')

    def _is_session_alive(self, session_id: str) -> bool:
        """Check if a session is still alive via state proxy.

        Args:
            session_id: Session ID to check

        Returns:
            True if session is alive, False if expired or not found
        """
        if not session_id:
            return True  # No session association means always alive

        # Get session last heartbeat through proxy
        last_heartbeat = self.state.get_session_last_heartbeat(session_id)
        if last_heartbeat is None:
            return False  # Session doesn't exist

        # Check if session has timed out using adapter_timeout
        return (time.time() - last_heartbeat) < self._adapter_timeout

    def unregister_adapter(self, adapter_name: str) -> bool:
        """Unregister an adapter from lifecycle tracking.

        Args:
            adapter_name: Name of the adapter to unregister.

        Returns:
            True if adapter was found and removed, False otherwise.
        """
        if adapter_name in self._adapter_records:
            adapter_info = self._adapter_records.pop(adapter_name)
            token = adapter_info.get('token')
            logger.debug(
                f"[AdapterManager] Unregistered adapter {adapter_name} for token {token[:8] if token else 'unknown'}..."
            )
            return True
        return False

    def set_adapter_state(self, adapter_name: str, key: str, value: Any) -> None:
        """Set a per-adapter state value.

        This is intentionally generic so higher-level services can store
        adapter-scoped state (e.g., training readiness) without maintaining
        separate side maps.
        """
        info = self._adapter_records.get(adapter_name)
        if info is None:
            return
        state = info.setdefault('state', {})
        state[key] = value

    def get_adapter_state(self, adapter_name: str, key: str, default: Any = None) -> Any:
        """Get a per-adapter state value."""
        info = self._adapter_records.get(adapter_name)
        if info is None:
            return default
        state = info.get('state') or {}
        return state.get(key, default)

    def pop_adapter_state(self, adapter_name: str, key: str, default: Any = None) -> Any:
        """Pop a per-adapter state value."""
        info = self._adapter_records.get(adapter_name)
        if info is None:
            return default
        state = info.get('state')
        if not isinstance(state, dict):
            return default
        return state.pop(key, default)

    def clear_adapter_state(self, adapter_name: str) -> None:
        """Clear all per-adapter state values."""
        info = self._adapter_records.get(adapter_name)
        if info is None:
            return
        info['state'] = {}

    def get_adapter_info(self, adapter_name: str) -> dict[str, Any] | None:
        """Get information about a registered adapter.

        Args:
            adapter_name: Name of the adapter to query.

        Returns:
            Dict with adapter information or None if not found.
        """
        return self._adapter_records.get(adapter_name)

    def _on_adapter_expired(self, adapter_name: str) -> None:
        """Hook method called when an adapter expires.

        This method must be overridden by inheriting classes to handle
        adapter expiration logic. The base implementation raises NotImplementedError.

        Args:
            adapter_name: Name of the expired adapter.

        Raises:
            NotImplementedError: If not overridden by inheriting class.
        """
        raise NotImplementedError(f'_on_adapter_expired must be implemented by {self.__class__.__name__}')

    @staticmethod
    def get_adapter_name(adapter_name: str) -> str:
        """Get the adapter name for a request.

        This is a passthrough method for consistency with the original API.

        Args:
            adapter_name: The adapter name (typically model_id)

        Returns:
            The adapter name to use
        """
        return adapter_name

    def assert_adapter_exists(self, adapter_name: str) -> None:
        """Validate that an adapter exists and is not expiring."""
        info = self._adapter_records.get(adapter_name)
        assert adapter_name and info is not None and not info.get('expiring'), \
            f'Adapter {adapter_name} not found'

    def _adapter_countdown_loop(self) -> None:
        """Background thread that monitors and handles adapters whose session has expired or exceeded max lifetime.

        This thread runs continuously and:
        1. Checks whether an adapter has exceeded `_adapter_max_lifetime` (sync, no async call)
        2. Checks session liveness for remaining adapters every second
        3. Calls _on_adapter_expired() for adapters that have expired
        4. Removes expired adapters from tracking
        """
        logger.debug(f'[AdapterManager] Countdown thread started (session_timeout={self._adapter_timeout}s)')
        while self._adapter_countdown_running:
            try:
                time.sleep(10)

                expired_adapters: list[tuple[str, str | None]] = []
                # Create snapshot to avoid modification during iteration
                adapter_snapshot = list(self._adapter_records.items())
                for adapter_name, info in adapter_snapshot:
                    if info.get('expiring'):
                        continue

                    session_id = info.get('session_id')
                    created_at = info.get('created_at', 0.0)
                    now = time.time()

                    # Check max lifetime first (no async call needed)
                    if now - created_at >= self._adapter_max_lifetime:
                        logger.debug(f'[AdapterManager] Adapter {adapter_name} exceeded max lifetime '
                                     f'({self._adapter_max_lifetime}s), marking as expired')
                        info['expiring'] = True
                        info['state'] = {}
                        token = info.get('token')
                        expired_adapters.append((adapter_name, token, session_id))
                        continue

                    try:
                        session_alive = self._is_session_alive(session_id)
                    except Exception as e:
                        logger.warning(f'[AdapterManager] Failed to check session liveness for {adapter_name}: '
                                       f'{type(e).__name__}: {e}')
                        continue
                    session_expired = not session_alive
                    logger.debug(f'[AdapterManager] Adapter {adapter_name} session check '
                                 f'(session_id={session_id}, session_alive={not session_expired})')

                    if session_expired:
                        info['expiring'] = True
                        info['state'] = {}  # best-effort clear
                        token = info.get('token')
                        expired_adapters.append((adapter_name, token, session_id))

                for adapter_name, _token, session_id in expired_adapters:
                    success = False
                    try:
                        self._on_adapter_expired(adapter_name)
                        logger.info(f'[AdapterManager] Adapter {adapter_name} expired '
                                    f'(reason=session_expired, session={session_id})')
                        success = True
                    except Exception as e:
                        logger.warning(f'[AdapterManager] Error while expiring adapter {adapter_name}: {e}')
                    finally:
                        if success:
                            self._adapter_records.pop(adapter_name, None)
                        else:
                            info = self._adapter_records.get(adapter_name)
                            if info is not None:
                                info['expiring'] = False

            except Exception as e:
                logger.warning(f'[AdapterManager] Error in countdown loop: {e}')
                continue

        logger.debug('[AdapterManager] Countdown thread stopped')

    def start_adapter_countdown(self) -> None:
        """Start the background adapter countdown thread.

        This should be called once when the mixin is initialized.
        It's safe to call multiple times - subsequent calls are ignored.
        """
        if not self._adapter_countdown_running:
            self._adapter_countdown_running = True
            self._adapter_countdown_thread = threading.Thread(target=self._adapter_countdown_loop, daemon=True)
            self._adapter_countdown_thread.start()
            logger.debug('[AdapterManager] Countdown thread started')

    def stop_adapter_countdown(self) -> None:
        """Stop the background adapter countdown thread.

        This should be called when shutting down the server.
        """
        if self._adapter_countdown_running:
            self._adapter_countdown_running = False
            if self._adapter_countdown_thread:
                # Wait for thread to finish (it checks the flag every second)
                self._adapter_countdown_thread.join(timeout=2.0)
            logger.debug('[AdapterManager] Countdown thread stopped')
