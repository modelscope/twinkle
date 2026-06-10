from __future__ import annotations

import asyncio
import fcntl
import json
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend


class FileBackend(StateBackend):
    """Local JSON file-based persistent state backend.

    Storage format is a single JSON file:
    ``{key: {"value": ..., "expire_at": float|null}}``.
    File I/O is wrapped with ``asyncio.to_thread`` to avoid blocking the
    event loop. Every operation that reads-or-writes goes through a sibling
    ``.lock`` file held with ``fcntl.LOCK_EX`` so concurrent processes and
    coroutines all serialize on the same critical section — that is the only
    way ``update_atomic`` can give a meaningful atomicity guarantee against a
    concurrent ``set`` / ``delete`` on the same key.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._lock_path = f'{file_path}.lock'
        self._init_file()

    def _init_file(self) -> None:
        """Auto-create the data file, the lock file, and any missing parent dir."""
        dir_path = os.path.dirname(self._file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self._file_path):
            with open(self._file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)
        # Touch the lock file so flock has a stable inode across processes.
        if not os.path.exists(self._lock_path):
            with open(self._lock_path, 'a', encoding='utf-8'):
                pass

    # ----- lock + file primitives ---------------------------------------- #

    @contextmanager
    def _locked(self):
        """Hold an exclusive flock on the sibling lock file for the block."""
        with open(self._lock_path, 'a+', encoding='utf-8') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _load_sync(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self._file_path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_sync(self, data: dict[str, dict[str, Any]]) -> None:
        """Write temp file then atomic-replace. Caller must hold ``_locked``."""
        # Drop expired entries on the write path so the file never grows
        # unbounded with stale keys.
        now = time.time()
        data = {k: v for k, v in data.items() if v.get('expire_at') is None or v['expire_at'] > now}

        dir_path = os.path.dirname(self._file_path) or '.'
        fd = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.tmp',
            dir=dir_path,
            delete=False,
            encoding='utf-8',
        )
        try:
            json.dump(data, fd, ensure_ascii=False)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
            os.replace(fd.name, self._file_path)
        except BaseException:
            if os.path.exists(fd.name):
                os.unlink(fd.name)
            raise

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        expire_at = entry.get('expire_at')
        return expire_at is not None and time.time() >= expire_at

    # ----- public API: every op runs under one lock ---------------------- #

    def _set_sync(self, key: str, value: Any, ttl: int | None) -> None:
        with self._locked():
            data = self._load_sync()
            expire_at = (time.time() + ttl) if ttl is not None else None
            data[key] = {'value': value, 'expire_at': expire_at}
            self._save_sync(data)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await asyncio.to_thread(self._set_sync, key, value, ttl)

    def _get_sync(self, key: str) -> Any | None:
        with self._locked():
            data = self._load_sync()
            entry = data.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del data[key]
                self._save_sync(data)
                return None
            return entry['value']

    async def get(self, key: str) -> Any | None:
        return await asyncio.to_thread(self._get_sync, key)

    def _delete_sync(self, key: str) -> None:
        with self._locked():
            data = self._load_sync()
            if key in data:
                del data[key]
                self._save_sync(data)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._delete_sync, key)

    def _exists_sync(self, key: str) -> bool:
        with self._locked():
            data = self._load_sync()
            entry = data.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del data[key]
                self._save_sync(data)
                return False
            return True

    async def exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, key)

    def _keys_sync(self, pattern: str) -> list[str]:
        with self._locked():
            data = self._load_sync()
            result: list[str] = []
            expired_keys: list[str] = []
            for key, entry in data.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
                    continue
                if fnmatch(key, pattern):
                    result.append(key)
            if expired_keys:
                for key in expired_keys:
                    del data[key]
                self._save_sync(data)
            return result

    async def keys(self, pattern: str) -> list[str]:
        return await asyncio.to_thread(self._keys_sync, pattern)

    async def count(self, pattern: str) -> int:
        return len(await self.keys(pattern))

    def _set_nx_sync(self, key: str, value: Any, ttl: int | None) -> bool:
        with self._locked():
            data = self._load_sync()
            entry = data.get(key)
            if entry is not None and not self._is_expired(entry):
                return False
            expire_at = (time.time() + ttl) if ttl is not None else None
            data[key] = {'value': value, 'expire_at': expire_at}
            self._save_sync(data)
            return True

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        return await asyncio.to_thread(self._set_nx_sync, key, value, ttl)

    def _update_atomic_sync(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None,
    ) -> Any | None:
        with self._locked():
            data = self._load_sync()
            entry = data.get(key)
            current = None if (entry is None or self._is_expired(entry)) else entry['value']
            new_value = transform(current)
            if new_value is None:
                return current
            expire_at = (time.time() + ttl) if ttl is not None else None
            data[key] = {'value': new_value, 'expire_at': expire_at}
            self._save_sync(data)
            return new_value

    async def update_atomic(
        self,
        key: str,
        transform: Callable[[Any | None], Any | None],
        ttl: int | None = None,
    ) -> Any | None:
        return await asyncio.to_thread(self._update_atomic_sync, key, transform, ttl)

    def _mget_sync(self, keys: list[str]) -> list[Any | None]:
        with self._locked():
            data = self._load_sync()
            results: list[Any | None] = []
            for key in keys:
                entry = data.get(key)
                if entry is None or self._is_expired(entry):
                    results.append(None)
                else:
                    results.append(entry['value'])
            return results

    async def mget(self, keys: list[str]) -> list[Any | None]:
        return await asyncio.to_thread(self._mget_sync, keys)

    async def close(self) -> None:
        """File backend has no persistent connection — nothing to release."""
        pass

    async def health_check(self) -> bool:
        try:
            return os.access(self._file_path, os.W_OK)
        except OSError:
            return False
