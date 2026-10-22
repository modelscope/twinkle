# Copyright (c) ModelScope Contributors. All rights reserved.
"""Resource-count publishing for the cleanup leader.

This is the single file in ``state/`` that imports the metrics registry.
Pulling ``_metrics_publish_loop`` out of ``ServerState`` shrinks the persistence
layer's dependency on the observability layer to this one module.

Only the cleanup leader publishes: ``ServerState`` drives this through the
coordinator's ``on_become_leader`` / ``on_lose_leader`` callbacks, so four Ray Serve
workers do not multiply the gauges --
``test_lgtm_telemetry.py::test_active_sessions_no_4x_inflation`` guards it.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from twinkle.server.telemetry import MetricsRegistry
from twinkle.utils.logger import get_logger
from .base import BaseManager

logger = get_logger()


class ResourceCountPublisher:
    """Owns the periodic push of resource counts into the ``MetricsRegistry`` cache.

    The ObservableGauges registered by :class:`MetricsRegistry` read the cache at OTEL
    export time and report whatever was pushed last, so this loop is the single writer
    of those four gauges.
    """

    def __init__(self, managers: Sequence[tuple[str, BaseManager]], *, interval: float) -> None:
        self._managers = tuple(managers)
        self._interval = float(interval)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start pushing counts. Idempotent — a second call while running is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def clear(self) -> None:
        """Zero this worker's resource-gauge cache.

        Called on leadership loss: after the publish loop is cancelled it never
        overwrites the cache again, so without zeroing the stale worker would keep
        emitting its last counts forever. The new leader publishes the authoritative
        counts from its own process.
        """
        MetricsRegistry.get().clear_resource_counts()

    async def _publish_loop(self) -> None:
        """Push resource counts into the MetricsRegistry cache every N seconds."""
        registry = MetricsRegistry.get()
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                for name, mgr in self._managers:
                    registry.set_resource_count(name, await mgr.count())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f'[ResourceCountPublisher] Error publishing metrics: {e}')
                continue
