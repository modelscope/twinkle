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

from fastapi import HTTPException
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
        self.register_resource(adapter_name, token, session_id)

    def unregister_adapter(self, adapter_name: str) -> bool:
        """Unregister an adapter from lifecycle tracking.

        Args:
            adapter_name: Name of the adapter to unregister.

        Returns:
            True if adapter was found and removed, False otherwise.
        """
        return self.unregister_resource(adapter_name)

    def set_adapter_state(self, adapter_name: str, key: str, value: Any) -> None:
        """Set a per-adapter state value.

        This is intentionally generic so higher-level services can store
        adapter-scoped state (e.g., training readiness) without maintaining
        separate side maps.
        """
        self.set_resource_state(adapter_name, key, value)

    def get_adapter_state(self, adapter_name: str, key: str, default: Any = None) -> Any:
        """Get a per-adapter state value."""
        return self.get_resource_state(adapter_name, key, default)

    def pop_adapter_state(self, adapter_name: str, key: str, default: Any = None) -> Any:
        """Pop a per-adapter state value."""
        return self.pop_resource_state(adapter_name, key, default)

    def clear_adapter_state(self, adapter_name: str) -> None:
        """Clear all per-adapter state values."""
        self.clear_resource_state(adapter_name)

    def get_adapter_info(self, adapter_name: str) -> dict[str, Any] | None:
        """Get information about a registered adapter.

        Args:
            adapter_name: Name of the adapter to query.

        Returns:
            Dict with adapter information or None if not found.
        """
        return self.get_resource_info(adapter_name)

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

    def assert_adapter_exists(self, adapter_name: str) -> None:
        """Validate that an adapter exists and is not expiring.

        Raises:
            HTTPException: 400 if adapter not found or expiring, with clear error message.
        """
        info = self._resource_records.get(adapter_name)
        if not adapter_name or info is None or info.get('expiring'):
            raise HTTPException(
                status_code=400,
                detail=f"Adapter '{adapter_name}' not found. "
                f'Please call add_adapter_to_model() first to create an adapter.')

    def _ensure_countdown_started(self) -> None:
        """Ensure the countdown task is started. Call from async context."""
        super()._ensure_countdown_started()

    def stop_adapter_countdown(self) -> None:
        """Stop the background countdown task."""
        self.stop_resource_countdown()
