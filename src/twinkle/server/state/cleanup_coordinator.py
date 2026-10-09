# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cleanup orchestration and cleanup-leader election.

Extracted from ``ServerState`` because these ~120 lines need only the
managers' ``count()`` / concrete ``cleanup_expired`` plus the shared
``StateBackend``, yet used to live beside ~25 CRUD facade methods, so any change
here meant re-reading all ~660 lines to confirm nothing else was touched.

Leader election lives here rather than in ``ServerState`` because it exists *for*
the cleanup loop: exactly one holder of the lease runs the cascade, so four Ray
Serve workers do not each sweep (and do not each publish resource counts). This
module deliberately does NOT import ``telemetry``: the resource-count publishing
is driven through the ``on_become_leader`` / ``on_lose_leader`` callbacks so the
telemetry dependency stays isolated to ``count_publisher``.
"""
from __future__ import annotations

import asyncio
import functools
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from twinkle.utils.logger import get_logger
from .backend import StateBackend
from .base import BaseManager

logger = get_logger()

# ---------- Cleanup-leader election ------------------------------------------
#
# Every Ray Serve worker creates its own ``ServerState``; without coordination
# each one would run the periodic cleanup and metrics-publish loop, so a single
# Twinkle deployment would multiply the work and inflate every gauge by the
# worker count. We elect one leader per backend by racing for a TTL-scoped key
# inside the shared StateBackend: the winner runs cleanup + publishes metrics,
# the others stay quiet.

LEADER_KEY = 'cleanup_leader'  # actual backend key: '<key_prefix>cleanup_leader'
LEASE_TTL = 30  # seconds — leader loses the lease after this without a renew
LEASE_RENEW = 10  # seconds — must be < LEASE_TTL/2 so two missed renews still beat the TTL


def _renew_if_owner(current: str | None, *, owner: str) -> str | None:
    """``update_atomic`` transform: only re-write the lease if it is still mine."""
    if current == owner:
        return owner
    return None


class ResourceCleanupCoordinator:
    """Owns the cleanup loop, the cascaded expiry across managers, and leader election."""

    def __init__(
        self,
        backend: StateBackend,
        managers: Mapping[str, BaseManager],
        *,
        cleanup_interval: float,
        expiration_timeout: float,
        sweep_processor_quotas: Callable[[], Awaitable[None]],
        on_become_leader: Callable[[], Awaitable[None]] | None = None,
        on_lose_leader: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._backend = backend
        self._managers = dict(managers)
        self._cleanup_interval = float(cleanup_interval)
        self._expiration_timeout = float(expiration_timeout)
        self._sweep_processor_quotas = sweep_processor_quotas
        self._on_become_leader = on_become_leader
        self._on_lose_leader = on_lose_leader

        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_running = False
        self._leader_id = uuid.uuid4().hex
        self._is_leader = False
        self._leader_task: asyncio.Task | None = None
        self._leader_running = False

    @property
    def is_leader(self) -> bool:
        return self._is_leader

    # ----- Resource cleanup -----

    async def cleanup_expired_resources(self) -> dict[str, int]:
        """Clean up expired sessions, models, sampling_sessions, and futures.

        Sessions expire based on last_heartbeat (or created_at). Models and sampling
        sessions are also cascade-expired when their owning session expires. Futures
        expire based on updated_at (or created_at).
        """
        current_time = time.time()
        cutoff_time = current_time - self._expiration_timeout

        session_mgr = self._managers['sessions']
        model_mgr = self._managers['models']
        sampling_mgr = self._managers['sampling_sessions']
        future_mgr = self._managers['futures']

        # Determine expired sessions and remove them in a SINGLE pass, then cascade
        # the SAME set to dependent resources. Using one authoritative set closes the
        # TOCTOU window where a session touched mid-cleanup could survive removal while
        # its children were cascade-deleted.
        expired_session_ids, sessions_removed = await session_mgr.collect_and_remove_expired(cutoff_time)

        models_removed = await model_mgr.cleanup_expired(cutoff_time, expired_session_ids=expired_session_ids)
        samplings_removed = await sampling_mgr.cleanup_expired(cutoff_time, expired_session_ids=expired_session_ids)

        alive_replica_ids = await model_mgr.get_alive_replica_ids(self._expiration_timeout)
        futures_removed = await future_mgr.cleanup_expired(cutoff_time, alive_replica_ids=alive_replica_ids)
        await self._sweep_processor_quotas()

        return {
            'sessions': sessions_removed,
            'models': models_removed,
            'sampling_sessions': samplings_removed,
            'futures': futures_removed,
        }

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up expired resources.

        Gated by leader election — non-leader workers skip the actual cleanup so the
        same backend isn't swept 4x by 4 deployment workers.
        """
        while self._cleanup_running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                if not self._is_leader:
                    continue
                stats = await self.cleanup_expired_resources()
                if any(stats.values()):
                    logger.debug(f'[ServerState Cleanup] Removed expired resources: {stats}')
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f'[ServerState Cleanup] Error during cleanup: {e}')
                continue

    # ----- Leader election -----

    async def _leader_loop(self) -> None:
        """Acquire and renew the cleanup-leader lease every LEASE_RENEW seconds."""
        await self._try_acquire_or_renew()  # Race for leadership at startup
        while self._leader_running:
            try:
                await asyncio.sleep(LEASE_RENEW)
                await self._try_acquire_or_renew()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f'[ServerState Leader] renew error: {e}')
                continue

    async def _try_acquire_or_renew(self) -> None:
        was_leader = self._is_leader
        try:
            if self._is_leader:
                val = await self._backend.update_atomic(
                    LEADER_KEY,
                    functools.partial(_renew_if_owner, owner=self._leader_id),
                    ttl=LEASE_TTL,
                )
                self._is_leader = (val == self._leader_id)
            else:
                self._is_leader = await self._backend.set_nx(LEADER_KEY, self._leader_id, ttl=LEASE_TTL)
        except Exception as e:
            logger.warning(f'[ServerState Leader] backend error during election: {e}')
            self._is_leader = False
            if was_leader:
                # Our renewal failed but our lease value may still be sitting in the
                # backend, so a plain ``set_nx`` would keep returning False for up to
                # LEASE_TTL and leadership would stall unclaimed. Best-effort delete
                # ONLY when we were the leader (never steal a lease another replica
                # legitimately holds), swallowing errors so a delete failure cannot
                # escape the election loop. The next tick can then re-acquire.
                try:
                    await self._backend.delete(LEADER_KEY)
                except Exception:
                    pass

        if self._is_leader and not was_leader:
            logger.info(f'[ServerState] became cleanup leader (id={self._leader_id[:8]})')
            if self._on_become_leader is not None:
                await self._on_become_leader()
        elif not self._is_leader and was_leader:
            logger.warning(f'[ServerState] lost cleanup leadership (id={self._leader_id[:8]})')
            if self._on_lose_leader is not None:
                await self._on_lose_leader()

    # ----- Lifecycle -----

    async def start(self) -> bool:
        """Start the background cleanup + leader-election tasks.

        Idempotent: returns ``False`` if already running. The guard lives here (not in
        ``ServerState``) so the implementation and its guard can never drift apart into
        a double-start.
        """
        if self._cleanup_running:
            return False
        # Rebuild in-memory indexes from backend data before the loops start.
        await self._managers['models'].rebuild_indexes()
        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._leader_running = True
        self._leader_task = asyncio.create_task(self._leader_loop())
        return True

    async def stop(self) -> bool:
        """Stop the background cleanup + leader-election tasks. Returns ``False`` if not running."""
        if not self._cleanup_running:
            return False
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        self._leader_running = False
        if self._leader_task:
            self._leader_task.cancel()
            self._leader_task = None
        if self._is_leader:
            # Release callback registration; the lease itself expires on its own TTL —
            # update_atomic can't express "atomic delete", so we accept a short outage
            # where the gauge reads 0 between leaders.
            if self._on_lose_leader is not None:
                await self._on_lose_leader()
            self._is_leader = False
        return True

    async def get_cleanup_stats(self) -> dict[str, Any]:
        """Get current cleanup configuration and resource counts."""
        return {
            'expiration_timeout': self._expiration_timeout,
            'cleanup_interval': self._cleanup_interval,
            'cleanup_running': self._cleanup_running,
            'is_leader': self._is_leader,
            'leader_id': self._leader_id,
            'resource_counts': {
                'sessions': await self._managers['sessions'].count(),
                'models': await self._managers['models'].count(),
                'sampling_sessions': await self._managers['sampling_sessions'].count(),
                'futures': await self._managers['futures'].count(),
            },
        }
