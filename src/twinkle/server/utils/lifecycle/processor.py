# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Processor Lifecycle Manager Mixin for Twinkle Server.

Mirrors AdapterManagerMixin but adds a global per-token processor limit.
Sessions are tracked via session ID; processors expire when their session expires.
"""
from __future__ import annotations

import time
from typing import Any

from twinkle.utils.logger import get_logger
from .base import SessionResourceMixin

logger = get_logger()


class ProcessorManagerMixin(SessionResourceMixin):
    """Mixin for processor lifecycle management with session-based expiration.

    Mirrors AdapterManagerMixin with an additional per-token processor limit.

    Inheriting classes should:
    1. Call _init_processor_manager() in __init__
    2. Override _on_processor_expired() to handle cleanup

    Attributes:
        _processor_timeout: Session inactivity timeout in seconds.
        _per_token_processor_limit: Maximum active processors per user token.
    """

    # Set resource type for logging
    _resource_type = 'Processor'

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
        self._init_resource_manager(
            resource_timeout=processor_timeout,
            resource_max_lifetime=None,  # No max lifetime for processors
        )
        self._per_token_processor_limit = per_token_processor_limit

    @property
    def _processor_timeout(self) -> float:
        """Processor timeout for backward compatibility."""
        return self._resource_timeout

    @property
    def _processor_records(self) -> dict[str, dict[str, Any]]:
        """Processor records for backward compatibility."""
        return self._resource_records

    def _validate_registration(self, resource_id: str, token: str, session_id: str) -> None:
        """Validate before registering a processor. Checks per-token limit.

        Args:
            resource_id: Processor identifier
            token: User token
            session_id: Session ID

        Raises:
            ValueError: If session_id is empty.
            RuntimeError: If per-token limit is reached.
        """
        super()._validate_registration(resource_id, token, session_id)

        current_count = sum(1 for info in self._resource_records.values() if info.get('token') == token)
        if current_count >= self._per_token_processor_limit:
            raise RuntimeError(f'Per-user processor limit ({self._per_token_processor_limit}) reached '
                               f'for token {token[:8]}...')

    def _create_resource_record(self, token: str, session_id: str) -> dict[str, Any]:
        """Create a new processor record without state field."""
        return {
            'token': token,
            'session_id': session_id,
            'created_at': time.time(),
            'expiring': False,
        }

    async def _on_resource_expired(self, resource_id: str) -> None:
        """Internal hook called by base class. Delegates to _on_processor_expired."""
        self._on_processor_expired(resource_id)

    def _on_processor_expired(self, processor_id: str) -> None:
        """Hook called when a processor's session expires.

        Must be overridden by inheriting classes.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError(f'_on_processor_expired must be implemented by {self.__class__.__name__}')

    def stop_processor_countdown(self) -> None:
        """Stop the background countdown task."""
        self.stop_resource_countdown()
