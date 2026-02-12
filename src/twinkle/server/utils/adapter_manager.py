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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy
    from twinkle.model import TwinkleModel
    
from twinkle.utils.logger import get_logger

logger = get_logger()


class AdapterManagerMixin:
    """Mixin for adapter lifecycle management with automatic timeout.

    This mixin tracks adapter activity and automatically expires adapters
    that have been inactive for longer than the configured timeout period.

    Inheriting classes should:
    1. Have a `self.model` attribute for model operations
    2. Call _init_adapter_manager() in __init__
    3. Optionally override _on_adapter_expired() to customize expiration handling

    Attributes:
        _adapter_timeout: Timeout in seconds for inactive adapters.
        model: Model instance for adapter operations (must be set by inheriting class).
    """

    # Type hint for state attribute that inheriting classes must provide
    state: 'ServerStateProxy'
    model: 'TwinkleModel'
    
    def _init_adapter_manager(self, adapter_timeout: float = 1800.0, per_token_adapter_limit: int = 30) -> None:
        """Initialize the adapter manager.

        This should be called in the __init__ of the inheriting class.

        Args:
            adapter_timeout: Timeout in seconds for inactive adapters.
                Default is 1800.0 (30 minutes).
            per_token_adapter_limit: Maximum number of adapters per user token.
                Default is 30.
        """
        self._adapter_timeout = adapter_timeout
        self._per_token_adapter_limit = per_token_adapter_limit

        # Adapter lifecycle tracking
        # Dict mapping adapter_name -> {'token': str, 'last_activity': float, 'created_at': float, 'inactivity_counter': int}
        self._adapter_records: Dict[str, Dict[str, Any]] = {}
        # Track adapter count per token
        self._adapter_counts: Dict[str, int] = {}
        self._adapter_lock = threading.Lock()

        # Countdown thread
        self._adapter_countdown_thread: Optional[threading.Thread] = None
        self._adapter_countdown_running = False

    def register_adapter(self, adapter_name: str, token: str) -> None:
        """Register a new adapter for lifecycle tracking.

        Args:
            adapter_name: Name of the adapter to register.
            token: User token that owns this adapter.
        """
        with self._adapter_lock:
            current_time = time.time()
            self._adapter_records[adapter_name] = {
                'token': token,
                'last_activity': current_time,
                'created_at': current_time,
                'inactivity_counter': 0,
            }
            logger.debug(
                f"[AdapterManager] Registered adapter {adapter_name} for token {token[:8]}...")

    def unregister_adapter(self, adapter_name: str) -> bool:
        """Unregister an adapter from lifecycle tracking.

        Args:
            adapter_name: Name of the adapter to unregister.

        Returns:
            True if adapter was found and removed, False otherwise.
        """
        with self._adapter_lock:
            if adapter_name in self._adapter_records:
                adapter_info = self._adapter_records.pop(adapter_name)
                token = adapter_info.get('token')
                logger.debug(
                    f"[AdapterManager] Unregistered adapter {adapter_name} for token {token[:8] if token else 'unknown'}...")
                return True
            return False

    def touch_adapter(self, adapter_name: str) -> bool:
        """Update adapter activity timestamp to prevent timeout.

        Args:
            adapter_name: Name of the adapter to touch.

        Returns:
            True if adapter was found and touched, False otherwise.
        """
        with self._adapter_lock:
            if adapter_name in self._adapter_records:
                self._adapter_records[adapter_name]['last_activity'] = time.time(
                )
                self._adapter_records[adapter_name]['inactivity_counter'] = 0
                return True
            return False

    def get_adapter_info(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered adapter.

        Args:
            adapter_name: Name of the adapter to query.

        Returns:
            Dict with adapter information or None if not found.
        """
        with self._adapter_lock:
            return self._adapter_records.get(adapter_name)

    def list_adapters(self, token: Optional[str] = None) -> List[str]:
        """List all registered adapters, optionally filtered by token.

        Args:
            token: Optional user token to filter by.

        Returns:
            List of adapter names.
        """
        with self._adapter_lock:
            if token is None:
                return list(self._adapter_records.keys())
            return [
                name for name, info in self._adapter_records.items()
                if info.get('token') == token
            ]

    def _on_adapter_expired(self, adapter_name: str, token: str) -> None:
        """Hook method called when an adapter expires.

        Default implementation removes the adapter from the model and updates adapter count.
        This is called from the countdown thread, so be careful with blocking operations.

        Args:
            adapter_name: Name of the expired adapter.
            token: User token that owns this adapter.
        """
        try:
            # Remove adapter from model
            self.model.remove_adapter(adapter_name)
            logger.info(
                f"[AdapterManager] Removed expired adapter {adapter_name} for token {token[:8]}...")

            # Decrement adapter count
            self.check_adapter_limit(token, False)
        except Exception as e:
            logger.warning(
                f"[AdapterManager] Failed to remove expired adapter {adapter_name}: {e}")

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
        """Validate that an adapter exists.

        Args:
            adapter_name: The adapter name to check

        Raises:
            AssertionError: If adapter doesn't exist
        """
        assert adapter_name and self.get_adapter_info(adapter_name) is not None, \
            f"Adapter {adapter_name} not found"

    def assert_adapter_valid(self, adapter_name: Optional[str]) -> None:
        """Validate that an adapter name is valid.

        Args:
            adapter_name: The adapter name to validate (can be None or empty)

        Raises:
            AssertionError: If adapter name is invalid
        """
        assert (adapter_name is None or adapter_name == '' or
                self.get_adapter_info(adapter_name) is not None), \
            f"Adapter {adapter_name} is invalid"

    def _adapter_countdown_loop(self) -> None:
        """Background thread that monitors and handles inactive adapters.

        This thread runs continuously and:
        1. Increments inactivity counters for all adapters every second
        2. Calls _on_adapter_expired() for adapters that exceed timeout
        3. Removes expired adapters from tracking
        """
        logger.debug(
            f"[AdapterManager] Countdown thread started (timeout={self._adapter_timeout}s)")
        while self._adapter_countdown_running:
            try:
                time.sleep(1)

                # Find and process expired adapters
                expired_adapters = []
                with self._adapter_lock:
                    for adapter_name, info in list(self._adapter_records.items()):
                        # Increment inactivity counter
                        info['inactivity_counter'] = info.get(
                            'inactivity_counter', 0) + 1

                        # Check if adapter has timed out
                        if info['inactivity_counter'] > self._adapter_timeout:
                            token = info.get('token')
                            expired_adapters.append((adapter_name, token))
                            self._adapter_records.pop(adapter_name, None)
                            logger.debug(
                                f"[AdapterManager] Adapter {adapter_name} timed out after "
                                f"{info['inactivity_counter']}s of inactivity"
                            )

                # Call hook method outside the lock
                for adapter_name, token in expired_adapters:
                    try:
                        self._on_adapter_expired(adapter_name, token)
                    except Exception as e:
                        logger.warning(
                            f"[AdapterManager] Error in _on_adapter_expired() "
                            f"for {adapter_name}: {e}"
                        )

            except Exception as e:
                logger.warning(
                    f"[AdapterManager] Error in countdown loop: {e}")
                continue

        logger.debug("[AdapterManager] Countdown thread stopped")

    def start_adapter_countdown(self) -> None:
        """Start the background adapter countdown thread.

        This should be called once when the mixin is initialized.
        It's safe to call multiple times - subsequent calls are ignored.
        """
        if not self._adapter_countdown_running:
            self._adapter_countdown_running = True
            self._adapter_countdown_thread = threading.Thread(
                target=self._adapter_countdown_loop,
                daemon=True
            )
            self._adapter_countdown_thread.start()
            logger.debug("[AdapterManager] Countdown thread started")

    def stop_adapter_countdown(self) -> None:
        """Stop the background adapter countdown thread.

        This should be called when shutting down the server.
        """
        if self._adapter_countdown_running:
            self._adapter_countdown_running = False
            if self._adapter_countdown_thread:
                # Wait for thread to finish (it checks the flag every second)
                self._adapter_countdown_thread.join(timeout=2.0)
            logger.debug("[AdapterManager] Countdown thread stopped")

    def get_adapter_stats(self) -> Dict[str, Any]:
        """Get adapter manager statistics.

        Returns:
            Dict with registered adapter count and configuration.
        """
        with self._adapter_lock:
            return {
                'registered_adapters': len(self._adapter_records),
                'tracked_adapter_counts': len(self._adapter_counts),
                'countdown_running': self._adapter_countdown_running,
                'adapter_timeout_seconds': self._adapter_timeout,
                'per_token_adapter_limit': self._per_token_adapter_limit,
            }

    def check_adapter_limit(self, token: str, add: bool) -> Tuple[bool, Optional[str]]:
        """Check and update adapter count for a user token.

        This method enforces per-user adapter limits to prevent resource exhaustion.

        Args:
            token: User token to check/update.
            add: True to add an adapter (increment count), False to remove (decrement count).

        Returns:
            Tuple of (allowed: bool, reason: Optional[str]).
            If allowed is False, reason contains the explanation.
        """
        user_key = token + '_' + 'model_adapter'
        with self._adapter_lock:
            current_count = self.state.get_config(user_key) or 0

            if add:
                # Check if adding would exceed limit
                if current_count >= self._per_token_adapter_limit:
                    return False, f"Adapter limit exceeded: {current_count}/{self._per_token_adapter_limit} adapters"
                # Increment count in global state
                self.state.add_config(user_key, current_count + 1)
                return True, None
            else:
                # Decrement count in global state
                if current_count > 0:
                    current_count -= 1
                    self.state.add_config(user_key, current_count)
                if current_count <= 0:
                    self.state.pop_config(user_key)
                return True, None

    def get_adapter_count(self, token: str) -> int:
        """Get current adapter count for a user token.

        Args:
            token: User token to query.

        Returns:
            Current number of adapters for this token.
        """
        user_key = token + '_' + 'model_adapter'
        with self._adapter_lock:
            return self.state.get_config(user_key) or 0
