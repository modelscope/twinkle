# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Processor Lifecycle Manager Mixin for Twinkle Server.

Mirrors AdapterManagerMixin but adds a global per-token processor limit.
Sessions are tracked via session ID; processors expire when their session expires.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from twinkle.server.utils.state import ServerStateProxy

from twinkle.utils.logger import get_logger

logger = get_logger()


class ProcessorManagerMixin:
    """Mixin for processor lifecycle management with session-based expiration.

    Mirrors AdapterManagerMixin with an additional per-token processor limit.

    Inheriting classes should:
    1. Call _init_processor_manager() in __init__
    2. Override _on_processor_expired() to handle cleanup

    Attributes:
        _processor_timeout: Session inactivity timeout in seconds.
        _per_token_processor_limit: Maximum active processors per user token.
    """

    # Type hint for state attribute that inheriting classes must provide
    state: ServerStateProxy

    def _init_processor_manager(
        self,
        processor_timeout: float = 1800.0,
        per_token_processor_limit: int = 20,
    ) -> None:
        """Initialize the processor manager.

        Args:
            processor_timeout: Timeout in seconds to determine if a session is alive.
                Default is 1800.0 (30 minutes).
            per_token_processor_limit: Maximum active processors per user token.
                Default is 20.
        """
        self._processor_timeout = processor_timeout
        self._per_token_processor_limit = per_token_processor_limit

        # processor_id -> {'token': str, 'session_id': str, 'created_at': float, 'expiring': bool}
        self._processor_records: dict[str, dict[str, Any]] = {}

        self._processor_countdown_thread: threading.Thread | None = None
        self._processor_countdown_running = False

    def register_processor(self, processor_id: str, token: str, session_id: str) -> None:
        """Register a new processor for lifecycle tracking.

        Args:
            processor_id: Unique identifier of the processor.
            token: User token that owns this processor.
            session_id: Session ID to associate with this processor. Must be non-empty.

        Raises:
            ValueError: If session_id is None or empty.
            RuntimeError: If the per-token processor limit has been reached.
        """
        if not session_id:
            raise ValueError(f'session_id must be provided when registering processor {processor_id}')

        current_count = sum(1 for info in self._processor_records.values() if info.get('token') == token)
        if current_count >= self._per_token_processor_limit:
            raise RuntimeError(f'Per-user processor limit ({self._per_token_processor_limit}) reached '
                               f'for token {token[:8]}...')

        self._processor_records[processor_id] = {
            'token': token,
            'session_id': session_id,
            'created_at': time.time(),
            'expiring': False,
        }
        logger.debug(f'[ProcessorManager] Registered processor {processor_id} '
                     f'for token {token[:8]}... (session: {session_id})')

    def unregister_processor(self, processor_id: str) -> bool:
        """Unregister a processor from lifecycle tracking.

        Returns:
            True if found and removed, False otherwise.
        """
        if processor_id in self._processor_records:
            info = self._processor_records.pop(processor_id)
            token = info.get('token', '')
            logger.debug(f'[ProcessorManager] Unregistered processor {processor_id} '
                         f'for token {token[:8] if token else "unknown"}...')
            return True
        return False

    def get_processor_info(self, processor_id: str) -> dict[str, Any] | None:
        """Get tracking info for a registered processor, or None if not found."""
        return self._processor_records.get(processor_id)

    def assert_processor_exists(self, processor_id: str) -> None:
        """Assert a processor exists and is not expiring."""
        info = self._processor_records.get(processor_id)
        assert processor_id and info is not None and not info.get('expiring'), \
            f'Processor {processor_id} not found'

    def _on_processor_expired(self, processor_id: str) -> None:
        """Hook called when a processor's session expires.

        Must be overridden by inheriting classes.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError(f'_on_processor_expired must be implemented by {self.__class__.__name__}')

    def _is_session_alive(self, session_id: str) -> bool:
        """Check if a session is still alive via state proxy."""
        if not session_id:
            return True
        last_heartbeat = self.state.get_session_last_heartbeat(session_id)
        if last_heartbeat is None:
            return False
        return (time.time() - last_heartbeat) < self._processor_timeout

    def _processor_countdown_loop(self) -> None:
        """Background thread: checks session liveness and expires stale processors."""
        logger.debug(f'[ProcessorManager] Countdown thread started (session_timeout={self._processor_timeout}s)')
        while self._processor_countdown_running:
            try:
                time.sleep(1)

                expired: list[tuple[str, str | None]] = []
                for processor_id, info in list(self._processor_records.items()):
                    if info.get('expiring'):
                        continue
                    session_id = info.get('session_id')
                    try:
                        session_alive = self._is_session_alive(session_id)
                    except Exception as e:
                        logger.warning(f'[ProcessorManager] Failed to check session liveness '
                                       f'for {processor_id}: {type(e).__name__}: {e}')
                        continue

                    logger.debug(f'[ProcessorManager] Processor {processor_id} session check '
                                 f'(session_id={session_id}, session_alive={session_alive})')
                    if not session_alive:
                        info['expiring'] = True
                        expired.append((processor_id, session_id))

                for processor_id, session_id in expired:
                    success = False
                    try:
                        self._on_processor_expired(processor_id)
                        logger.info(f'[ProcessorManager] Processor {processor_id} expired '
                                    f'(reason=session_expired, session={session_id})')
                        success = True
                    except Exception as e:
                        logger.warning(f'[ProcessorManager] Error while expiring processor {processor_id}: {e}')
                    finally:
                        if success:
                            self._processor_records.pop(processor_id, None)
                        else:
                            info = self._processor_records.get(processor_id)
                            if info is not None:
                                info['expiring'] = False

            except Exception as e:
                logger.warning(f'[ProcessorManager] Error in countdown loop: {e}')
                continue

        logger.debug('[ProcessorManager] Countdown thread stopped')

    def start_processor_countdown(self) -> None:
        """Start the background countdown thread. Safe to call multiple times."""
        if not self._processor_countdown_running:
            self._processor_countdown_running = True
            self._processor_countdown_thread = threading.Thread(target=self._processor_countdown_loop, daemon=True)
            self._processor_countdown_thread.start()
            logger.debug('[ProcessorManager] Countdown thread started')

    def stop_processor_countdown(self) -> None:
        """Stop the background countdown thread."""
        if self._processor_countdown_running:
            self._processor_countdown_running = False
            if self._processor_countdown_thread:
                self._processor_countdown_thread.join(timeout=2.0)
            logger.debug('[ProcessorManager] Countdown thread stopped')
