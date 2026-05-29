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
    """基于本地 JSON 文件的持久化状态后端实现。

    存储格式为单个 JSON 文件：``{key: {"value": ..., "expire_at": float|null}}``。
    文件读写通过 ``asyncio.to_thread`` 包装，避免阻塞事件循环。
    写入使用临时文件 + ``os.replace`` 原子替换，并通过 ``fcntl.flock`` 防止多进程并发写入。
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._init_file()

    def _init_file(self) -> None:
        """如果文件或目录不存在，自动创建。"""
        dir_path = os.path.dirname(self._file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(self._file_path):
            with open(self._file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_sync(self) -> dict[str, dict[str, Any]]:
        """同步读取 JSON 文件，返回完整 data dict。"""
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
        return data

    def _save_sync(self, data: dict[str, dict[str, Any]]) -> None:
        """同步写入：清理过期键 → 写临时文件 → flock → os.replace。"""
        # 写入前清理过期键
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

            # 对临时文件加排他锁后原子替换
            with open(fd.name, 'r') as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                os.replace(fd.name, self._file_path)
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except BaseException:
            # 清理临时文件
            if os.path.exists(fd.name):
                os.unlink(fd.name)
            raise

    async def _load(self) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self._load_sync)

    async def _save(self, data: dict[str, dict[str, Any]]) -> None:
        await asyncio.to_thread(self._save_sync, data)

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """检查条目是否已过期。"""
        expire_at = entry.get('expire_at')
        return expire_at is not None and time.time() >= expire_at

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存储键值对，可选 TTL（秒）"""
        expire_at = (time.time() + ttl) if ttl is not None else None
        data = await self._load()
        data[key] = {'value': value, 'expire_at': expire_at}
        await self._save(data)

    async def get(self, key: str) -> Any | None:
        """获取值，不存在或已过期返回 None"""
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
        """删除键，不存在时静默忽略"""
        data = await self._load()
        if key in data:
            del data[key]
            await self._save(data)

    async def exists(self, key: str) -> bool:
        """检查键是否存在且未过期"""
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
        """按模式匹配返回所有键名。pattern 支持 * 通配符。"""
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
        """按模式匹配计数"""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. 返回 True 如果成功设置，False 如果键已存在且未过期。"""
        data = await self._load()
        entry = data.get(key)
        if entry is not None and not self._is_expired(entry):
            return False
        data[key] = {'value': value, 'expire_at': None}
        await self._save(data)
        return True

    async def close(self) -> None:
        """关闭后端，文件后端无需持久连接，no-op。"""
        pass

    async def health_check(self) -> bool:
        """检查文件路径是否可写"""
        try:
            return os.access(self._file_path, os.W_OK)
        except OSError:
            return False
