from __future__ import annotations

import time
from fnmatch import fnmatch
from typing import Any

from .base import StateBackend


class MemoryBackend(StateBackend):
    """基于内存字典的状态后端实现。

    使用 ``dict[str, tuple[Any, float | None]]`` 存储 (value, expire_at)。
    过期检查在 get/exists 时进行（惰性过期），适用于 Ray Actor 单线程模型。
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}

    def _is_expired(self, key: str) -> bool:
        """检查键是否已过期。如已过期则删除并返回 True。"""
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.time() >= expire_at:
            del self._store[key]
            return True
        return False

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存储键值对，可选 TTL（秒）"""
        expire_at = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (value, expire_at)

    async def get(self, key: str) -> Any | None:
        """获取值，不存在或已过期返回 None"""
        if self._is_expired(key):
            return None
        value, _ = self._store[key]
        return value

    async def delete(self, key: str) -> None:
        """删除键，不存在时静默忽略"""
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        return not self._is_expired(key)

    async def keys(self, pattern: str) -> list[str]:
        """按模式匹配返回所有键名。pattern 支持 * 通配符。"""
        result: list[str] = []
        # 遍历时收集过期键，避免在迭代中修改字典
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
        """按模式匹配计数"""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. 返回 True 如果成功设置，False 如果键已存在。"""
        if not self._is_expired(key):
            return False
        self._store[key] = (value, None)
        return True

    async def close(self) -> None:
        """关闭后端，清空存储"""
        self._store.clear()

    async def health_check(self) -> bool:
        """检查后端是否健康可用，内存后端始终返回 True"""
        return True
