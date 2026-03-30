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

from typing import Any

from twinkle.utils.logger import get_logger
from .base import SessionResourceMixin

logger = get_logger()


class AdapterManagerMixin(SessionResourceMixin):
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

    # Set resource type for logging
    _resource_type = 'Adapter'

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
        self._init_resource_manager(
            resource_timeout=adapter_timeout,
            resource_max_lifetime=adapter_max_lifetime,
        )

    @property
    def _adapter_timeout(self) -> float:
        """Adapter timeout for backward compatibility."""
        return self._resource_timeout

    @property
    def _adapter_max_lifetime(self) -> float | None:
        """Adapter max lifetime for backward compatibility."""
        return self._resource_max_lifetime

    @property
    def _adapter_records(self) -> dict[str, dict[str, Any]]:
        """Adapter records for backward compatibility."""
        return self._resource_records

    async def _on_resource_expired(self, resource_id: str) -> None:
        """Internal hook called by base class. Delegates to _on_adapter_expired."""
        await self._on_adapter_expired(resource_id)

    async def _on_adapter_expired(self, adapter_name: str) -> None:
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

    def stop_adapter_countdown(self) -> None:
        """Stop the background countdown task."""
        self.stop_resource_countdown()
