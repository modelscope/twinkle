# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Processor Lifecycle Manager Mixin for Twinkle Server.

Mirrors AdapterManagerMixin but adds a global per-token processor limit.
Sessions are tracked via session ID; processors expire when their session expires.
"""
from __future__ import annotations

from abc import abstractmethod

from twinkle.utils.logger import get_logger
from .base import SessionResourceMixin

logger = get_logger()


class ProcessorManagerMixin(SessionResourceMixin):
    """Mixin for processor lifecycle management with session-based expiration.

    Mirrors AdapterManagerMixin with an additional per-token processor limit.

    Inheriting classes should:
    1. Call _init_processor_manager() in __init__
    2. Override _on_processor_expired() to handle cleanup

    The inactivity timeout is stored on the base mixin as ``_resource_timeout``
    (set via ``_init_processor_manager``); ``_per_token_processor_limit`` caps the
    active processors per user token.
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
        # The countdown runs every 10 seconds. A 30-second lease tolerates two
        # missed renewals while still bounding stale reservations after a crash.
        self._processor_quota_lease_seconds = 30.0

    async def _on_resource_liveness_confirmed(self, resource_id: str) -> bool:
        """Renew this processor's shared quota lease after a healthy probe."""
        info = self._resource_records.get(resource_id)
        if info is None:
            return False
        try:
            renewed = await self.state.renew_processor_quota(
                info['token'],
                resource_id,
                lease_seconds=self._processor_quota_lease_seconds,
            )
        except Exception as exc:
            # Keep the local processor during a transient backend outage. Once the
            # backend recovers, a lost/expired reservation returns False and the
            # countdown loop removes the unaccounted local resource.
            logger.warning('[ProcessorManager] Failed to renew quota lease for %s: %r', resource_id, exc)
            return True
        if not renewed:
            logger.warning('[ProcessorManager] Quota lease for %s was lost; expiring local processor', resource_id)
        return renewed

    async def _on_resource_expired(self, resource_id: str) -> None:
        """Base-class expiry hook; forwards to the domain hook ``_on_processor_expired``.

        ``_on_processor_expired`` is the supported extension point: the
        processor-domain name is kept deliberately so subclass authors override a
        method named for processors rather than the generic base-class hook. It is
        ``async`` to match the sibling ``AdapterManagerMixin._on_adapter_expired``
        contract, so both resource kinds expose the same extension-point shape.
        """
        await self._on_processor_expired(resource_id)

    @abstractmethod
    async def _on_processor_expired(self, processor_id: str) -> None:
        """Hook called when a processor's session expires."""
        ...
