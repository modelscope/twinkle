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
    """Mixin for adapter lifecycle management with automatic timeout.

    This mixin tracks adapter activity and automatically expires adapters
    that have been inactive for longer than the configured timeout period.

    Inheriting classes should:
    1. Call _init_adapter_manager() in __init__
    2. Override _on_adapter_expired() to customize expiration handling

    Attributes:
        _adapter_timeout: Timeout in seconds for inactive adapters.
    """

    # Type hint for state attribute that inheriting classes must provide
    state: ServerStateProxy

    def _init_adapter_manager(
        self,
        adapter_timeout: float = 1800.0,
        per_token_adapter_limit: int = 30,
        adapter_max_lifetime: float = 12 * 60 * 60,
    ) -> None:
        """Initialize the adapter manager.

        This should be called in the __init__ of the inheriting class.

        Args:
            adapter_timeout: Timeout in seconds for inactive adapters and session-based expiration.
                Default is 1800.0 (30 minutes). Adapters linked to sessions will expire
                when their session hasn't been touched for this duration.
            per_token_adapter_limit: Maximum number of adapters per user token.
                Default is 30.
            adapter_max_lifetime: Maximum lifetime in seconds for an adapter since creation.
                Default is 43200.0 (12 hours). If <= 0, lifetime enforcement is disabled.
        """
        self._adapter_timeout = adapter_timeout
        self._per_token_adapter_limit = per_token_adapter_limit
        self._adapter_max_lifetime = adapter_max_lifetime

        # Adapter lifecycle tracking
        # Dict mapping adapter_name ->
        # {'token': str, 'session_id': str, 'last_activity': float, 'created_at': float, 'inactivity_counter': int}
        self._adapter_records: dict[str, dict[str, Any]] = {}
        # Track adapter count per token
        self._adapter_counts: dict[str, int] = {}

        # Countdown thread
        self._adapter_countdown_thread: threading.Thread | None = None
        self._adapter_countdown_running = False

    def register_adapter(self, adapter_name: str, token: str, session_id: str | None = None) -> None:
        """Register a new adapter for lifecycle tracking.

        Args:
            adapter_name: Name of the adapter to register.
            token: User token that owns this adapter.
            session_id: Optional session ID to associate with this adapter.
                If provided, adapter will expire when the session expires.

        Raises:
            RuntimeError: If adapter limit is exceeded for this token.
        """
        # Check adapter limit BEFORE registering
        allowed, reason = self.check_adapter_limit(token)
        if not allowed:
            raise RuntimeError(reason)

        current_time = time.time()
        self._adapter_records[adapter_name] = {
            'token': token,
            'session_id': session_id,
            'last_activity': current_time,
            'created_at': current_time,
            'inactivity_counter': 0,
            'state': {},
            'expiring': False,
        }
        logger.debug(f'[AdapterManager] Registered adapter {adapter_name} for token {token[:8]}...'
                     + (f' (session: {session_id})' if session_id else ''))

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

    def touch_adapter(self, adapter_name: str) -> bool:
        """Update adapter activity timestamp to prevent timeout.

        Args:
            adapter_name: Name of the adapter to touch.

        Returns:
            True if adapter was found and touched, False otherwise.
        """
        info = self._adapter_records.get(adapter_name)
        if not info:
            return False
        if info.get('expiring'):
            return False
        info['last_activity'] = time.time()
        info['inactivity_counter'] = 0
        return True

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
        """Background thread that monitors and handles inactive adapters.

        This thread runs continuously and:
        1. Increments inactivity counters for all adapters every second
        2. Calls _on_adapter_expired() for adapters that exceed timeout
        3. Removes expired adapters from tracking
        """
        logger.debug(f'[AdapterManager] Countdown thread started (timeout={self._adapter_timeout}s)')
        while self._adapter_countdown_running:
            try:
                time.sleep(1)
                now = time.time()

                expired_adapters: list[tuple[str, str | None]] = []
                # Create snapshot to avoid modification during iteration
                adapter_snapshot = list(self._adapter_records.items())
                for adapter_name, info in adapter_snapshot:
                    if info.get('expiring'):
                        continue

                    session_id = info.get('session_id')
                    created_at = info.get('created_at')

                    # Check TTL for both cases
                    exceeded_ttl = (
                        self._adapter_max_lifetime and self._adapter_max_lifetime > 0
                        and (now - created_at) > self._adapter_max_lifetime)

                    # Different logic based on session association
                    if session_id:
                        # Has session: check session expiration and TTL
                        session_expired = not self._is_session_alive(session_id)
                        should_expire = session_expired or exceeded_ttl
                        logger.debug(
                            f'[AdapterManager] Adapter {adapter_name} session expiration check '
                            f'(session_id={session_id}, session_alive={not session_expired}, should_expire={should_expire})'  # noqa:E501
                        )
                        expiration_reasons = []
                        if exceeded_ttl:
                            expiration_reasons.append('ttl_exceeded')
                        if session_expired:
                            expiration_reasons.append('session_expired')
                    else:
                        # No session: check inactivity timeout and TTL
                        info['inactivity_counter'] = info.get('inactivity_counter', 0) + 1
                        exceeded_inactivity = info['inactivity_counter'] > self._adapter_timeout
                        should_expire = exceeded_ttl or exceeded_inactivity
                        logger.debug(
                            f'[AdapterManager] Adapter {adapter_name} inactivity check '
                            f'(inactivity_counter={info["inactivity_counter"]}, timeout={self._adapter_timeout}, should_expire={should_expire})'  # noqa:E501
                        )
                        expiration_reasons = []
                        if exceeded_ttl:
                            expiration_reasons.append('ttl_exceeded')
                        if exceeded_inactivity:
                            expiration_reasons.append('inactivity_timeout')

                    if should_expire:
                        info['expiring'] = True
                        info['state'] = {}  # best-effort clear
                        token = info.get('token')
                        expired_adapters.append((adapter_name, token))

                for adapter_name, token in expired_adapters:
                    success = False
                    try:
                        self._on_adapter_expired(adapter_name)
                        logger.info(f'[AdapterManager] Adapter {adapter_name} expired '
                                    f"(reasons={','.join(expiration_reasons)}, session={session_id})")
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

    def check_adapter_limit(self, token: str) -> tuple[bool, str | None]:
        """Check adapter count for a user token.

        This method enforces per-user adapter limits to prevent resource exhaustion.
        Counts adapters directly from _adapter_records instead of using state storage.

        Args:
            token: User token to check.

        Returns:
            Tuple of (allowed: bool, reason: Optional[str]).
            If allowed is False, reason contains the explanation.
        """
        # Count adapters directly from _adapter_records
        current_count = sum(1 for record in self._adapter_records.values()
                            if record.get('token') == token and not record.get('expiring', False))

        # Check if current count exceeds limit
        if current_count >= self._per_token_adapter_limit:
            return False, f'Adapter limit exceeded: {current_count}/{self._per_token_adapter_limit} adapters'
        return True, None
