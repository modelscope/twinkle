from __future__ import annotations

import ray
import time
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend

_ACTOR_NAME_PREFIX = 'twinkle_state_actor'


def _actor_name(key_prefix: str) -> str:
    """Namespace the detached actor by ``key_prefix`` so multiple Twinkle
    deployments sharing one Ray cluster do not collide.
    """
    return _ACTOR_NAME_PREFIX if not key_prefix else f'{_ACTOR_NAME_PREFIX}::{key_prefix}'


@ray.remote
class _StateActor:
    """Single-threaded asyncio actor that owns the canonical in-memory store.

    Ray's actor concurrency model serializes calls into one asyncio loop, so
    every operation here is atomic against every other — ``set_nx`` and
    ``update_atomic`` cannot race against any other op on this actor.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.time() >= expire_at:
            del self._store[key]
            return True
        return False

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (value, expire_at)

    async def get(self, key: str) -> Any | None:
        if self._is_expired(key):
            return None
        value, _ = self._store[key]
        return value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return not self._is_expired(key)

    async def keys(self, pattern: str) -> list[str]:
        result: list[str] = []
        expired_keys: list[str] = []
        for key, (_, expire_at) in self._store.items():
            if expire_at is not None and time.time() >= expire_at:
                expired_keys.append(key)
                continue
            if fnmatch(key, pattern):
                result.append(key)
        for key in expired_keys:
            del self._store[key]
        return result

    async def count(self, pattern: str) -> int:
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        if not self._is_expired(key):
            return False
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (value, expire_at)
        return True

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        current = None if self._is_expired(key) else self._store[key][0]
        new_value = transform(current)
        if new_value is None:
            return current
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (new_value, expire_at)
        return new_value

    async def mget(self, keys: list[str]) -> list[Any | None]:
        results: list[Any | None] = []
        for key in keys:
            if self._is_expired(key):
                results.append(None)
            else:
                value, _ = self._store[key]
                results.append(value)
        return results

    async def close(self) -> None:
        self._store.clear()

    async def health_check(self) -> bool:
        return True


class RayActorBackend(StateBackend):
    """Cross-process state backend backed by a detached Ray actor.

    The canonical store lives inside a detached ``_StateActor`` and every
    method forwards as ``await actor.X.remote(...)`` so all Ray Serve workers
    share one consistent view. This backend requires an initialized Ray
    runtime — there is no fallback to a process-local dict. (The operator-facing
    persistence mode literal for this backend is still ``"memory"``.)
    """

    def __init__(self, key_prefix: str = '') -> None:
        if not ray.is_initialized():
            raise RuntimeError('RayActorBackend requires an initialized Ray runtime — call '
                               'ray.init() first, switch persistence to "file"/"redis", or '
                               'rely on the deployment launcher to start Ray.')
        name = _actor_name(key_prefix)
        try:
            self._actor = ray.get_actor(name)
        except ValueError:
            try:
                self._actor = _StateActor.options(name=name, lifetime='detached').remote()
            except ValueError:
                # Lost the create race against another worker — the actor is
                # now resolvable.
                self._actor = ray.get_actor(name)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._actor.set.remote(key, value, ttl)

    async def get(self, key: str) -> Any | None:
        return await self._actor.get.remote(key)

    async def delete(self, key: str) -> None:
        await self._actor.delete.remote(key)

    async def exists(self, key: str) -> bool:
        return await self._actor.exists.remote(key)

    async def keys(self, pattern: str) -> list[str]:
        return await self._actor.keys.remote(pattern)

    async def count(self, pattern: str) -> int:
        return await self._actor.count.remote(pattern)

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        return await self._actor.set_nx.remote(key, value, ttl)

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        return await self._actor.update_atomic.remote(key, transform, ttl)

    async def mget(self, keys: list[str]) -> list[Any | None]:
        return await self._actor.mget.remote(keys)

    async def close(self) -> None:
        await self._actor.close.remote()

    async def health_check(self) -> bool:
        try:
            return await self._actor.health_check.remote()
        except ray.exceptions.RayActorError:
            # The actor crashed (OOM, node died). Don't silently re-create
            # it — that would lose all in-memory state. Let readiness probes
            # see False and the deployment owner decide to restart.
            return False
