from __future__ import annotations

import ray
import time
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend

_ACTOR_NAME_PREFIX = 'twinkle_state_actor'

#: Protocol version of the detached state actor. Bump it whenever the actor's call contract changes
#: incompatibly; ``RayActorBackend`` replaces any resolved actor that does not report this version.
_STATE_PROTOCOL_VERSION = 1


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
        *,
        return_changed: bool = False,
    ) -> Any | None:
        current = None if self._is_expired(key) else self._store[key][0]
        new_value = transform(current)
        if new_value is None:
            return False if return_changed else current
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (new_value, expire_at)
        return True if return_changed else new_value

    async def mget(self, keys: list[str]) -> list[Any | None]:
        results: list[Any | None] = []
        for key in keys:
            if self._is_expired(key):
                results.append(None)
            else:
                value, _ = self._store[key]
                results.append(value)
        return results

    async def flush_all(self) -> None:
        """Destructively clear shared state for tests and explicit admin flows."""
        self._store.clear()

    async def health_check(self) -> bool:
        return True

    def protocol_version(self) -> int:
        """Compatibility handshake for ``RayActorBackend``.

        Takes no arguments and returns a plain int, so it succeeds even on a worker that cannot
        import the caller's modules -- an actor that predates this method (or cannot answer it)
        is from an incompatible twinkle version and must be replaced, not reused.
        """
        return _STATE_PROTOCOL_VERSION


class RayActorBackend(StateBackend):
    """Cross-process state backend backed by a detached Ray actor.

    The canonical store lives inside a detached ``_StateActor`` and every
    method forwards as ``await actor.X.remote(...)`` so all Ray Serve workers
    share one consistent view. This backend requires an initialized Ray
    runtime — there is no fallback to a process-local dict. (The operator-facing
    persistence mode literal for this backend is still ``"memory"``.)
    """

    def __init__(self, key_prefix: str = '', namespace: str | None = None) -> None:
        if not ray.is_initialized():
            raise RuntimeError('RayActorBackend requires an initialized Ray runtime — call '
                               'ray.init() first, switch persistence to "redis", or rely on '
                               'the deployment launcher to start Ray.')
        name = _actor_name(key_prefix)
        self._actor = self._resolve_compatible_actor(name, namespace)

    @classmethod
    def _resolve_compatible_actor(cls, name: str, namespace: str | None):
        """Resolve the detached state actor, replacing an incompatible leftover.

        The actor is detached and named, so it outlives every launcher -- including a twinkle
        upgrade in place. An actor started by an older version (e.g. before the package split)
        runs in a worker whose ``sys.path`` predates the new modules: it cannot deserialize the
        objects the current code sends, and EVERY state call fails with
        ``ModuleNotFoundError`` -> HTTP 500 while the actor still "exists" and answers pings.
        Reusing it by name alone is therefore unsafe; handshake on ``protocol_version`` (no
        arguments, plain-int reply, so even a foreign worker can answer it) and replace any actor
        that does not report the current version. Bounded retries cover the create/kill race
        between concurrent workers.
        """
        for _ in range(3):
            try:
                actor = ray.get_actor(name, namespace=namespace)
            except ValueError:
                try:
                    actor = _StateActor.options(
                        name=name,
                        namespace=namespace,
                        lifetime='detached',
                        max_restarts=-1 if namespace else 0,
                        max_task_retries=0).remote()
                except ValueError:
                    # Lost the create race against another worker — the actor is
                    # now resolvable.
                    actor = ray.get_actor(name, namespace=namespace)
            try:
                version = ray.get(actor.protocol_version.remote(), timeout=30)
            except Exception:
                version = None
            if version == _STATE_PROTOCOL_VERSION:
                return actor
            # Incompatible (or unresponsive) leftover: kill it WITHOUT restart (max_restarts=-1
            # would otherwise resurrect the same old code) and loop to create/resolve a fresh one.
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        raise RuntimeError(f'Could not resolve a compatible twinkle state actor {name!r} '
                           f'in namespace {namespace!r} after repeated replacement attempts.')

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
        *,
        return_changed: bool = False,
    ) -> Any | None:
        return await self._actor.update_atomic.remote(key, transform, ttl, return_changed=return_changed)

    async def mget(self, keys: list[str]) -> list[Any | None]:
        return await self._actor.mget.remote(keys)

    async def close(self) -> None:
        """Release this process's actor handle without clearing shared state."""
        self._actor = None

    async def health_check(self) -> bool:
        try:
            return bool(await self._actor.health_check.remote())
        except Exception:
            # The actor crashed (OOM, node died) or this local handle was closed.
            # Do not silently recreate it because that could hide state loss.
            return False
