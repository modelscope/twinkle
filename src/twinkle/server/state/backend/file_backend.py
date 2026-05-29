from __future__ import annotations

import asyncio
import fcntl
import json
import os
import tempfile
import time
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend


class FileBackend(StateBackend):
    """Local JSON file-based persistent state backend implementation.

    Storage format is a single JSON file: ``{key: {"value": ..., "expire_at": float|null}}``.
    File I/O is wrapped with ``asyncio.to_thread`` to avoid blocking the event loop.
    Writes use temp file + ``os.replace`` for atomic replacement, protected by ``fcntl.flock`` against multi-process concurrent writes.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._init_file()

    def _init_file(self) -> None:
        """Auto-create file or directory if not exists."""
        dir_path = os.path.dirname(self._file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self._file_path):
            with open(self._file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_sync(self) -> dict[str, dict[str, Any]]:
        """Synchronously read JSON file, return complete data dict."""
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
        return data

    def _save_sync(self, data: dict[str, dict[str, Any]]) -> None:
        """Synchronous write: clean expired keys -> write temp file -> flock -> os.replace."""
        # Clean expired keys before writing
        now = time.time()
        data = {
            k: v for k, v in data.items()
            if v.get('expire_at') is None or v['expire_at'] > now
        }

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

            # Apply exclusive lock to temp file then atomic replace
            with open(fd.name, 'r') as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                os.replace(fd.name, self._file_path)
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except BaseException:
            # Clean up temp file
            if os.path.exists(fd.name):
                os.unlink(fd.name)
            raise

    async def _load(self) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self._load_sync)

    async def _save(self, data: dict[str, dict[str, Any]]) -> None:
        await asyncio.to_thread(self._save_sync, data)

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if entry is expired."""
        expire_at = entry.get('expire_at')
        return expire_at is not None and time.time() >= expire_at

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store key-value pair with optional TTL in seconds."""
        expire_at = (time.time() + ttl) if ttl is not None else None
        data = await self._load()
        data[key] = {'value': value, 'expire_at': expire_at}
        await self._save(data)

    async def get(self, key: str) -> Any | None:
        """Retrieve value, return None if not found or expired."""
        data = await self._load()
        entry = data.get(key)
        if entry is None:
            return None
        if self._is_expired(entry):
            del data[key]
            await self._save(data)
            return None
        return entry['value']

    async def delete(self, key: str) -> None:
        """Delete key, silently ignore if not found."""
        data = await self._load()
        if key in data:
            del data[key]
            await self._save(data)

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        data = await self._load()
        entry = data.get(key)
        if entry is None:
            return False
        if self._is_expired(entry):
            del data[key]
            await self._save(data)
            return False
        return True

    async def keys(self, pattern: str) -> list[str]:
        """Return all key names matching the pattern. Supports * wildcard."""
        data = await self._load()
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
            await self._save(data)
        return result

    async def count(self, pattern: str) -> int:
        """Count keys matching the pattern."""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. Return True if successfully set, False if key already exists and is not expired."""
        data = await self._load()
        entry = data.get(key)
        if entry is not None and not self._is_expired(entry):
            return False
        data[key] = {'value': value, 'expire_at': None}
        await self._save(data)
        return True

    async def close(self) -> None:
        """Close backend. File backend requires no persistent connection, no-op."""
        pass

    async def health_check(self) -> bool:
        """Check if file path is writable."""
        try:
            return os.access(self._file_path, os.W_OK)
        except OSError:
            return False
