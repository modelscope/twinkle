from __future__ import annotations

import json
from typing import Any

from .base import StateBackend

try:
    import redis.asyncio as aioredis

    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisBackend(StateBackend):
    """基于 Redis 的持久化状态后端实现。

    使用 ``redis.asyncio`` 客户端，值通过 JSON 序列化存储为 Redis string。
    TTL 由 Redis 原生 EXPIRE 机制管理。
    """

    def __init__(self, redis_url: str, key_prefix: str = "") -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis package required. Install with: pip install redis"
            )
        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._prefix = key_prefix

    def _make_key(self, key: str) -> str:
        """为 key 添加命名空间前缀。"""
        return f"{self._prefix}{key}" if self._prefix else key

    def _strip_prefix(self, key: str) -> str:
        """从完整 key 中移除命名空间前缀。"""
        return key[len(self._prefix):] if self._prefix else key

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存储键值对，可选 TTL（秒）"""
        real_key = self._make_key(key)
        data = json.dumps(value)
        if ttl is not None:
            await self._client.set(real_key, data, ex=ttl)
        else:
            await self._client.set(real_key, data)

    async def get(self, key: str) -> Any | None:
        """获取值，不存在或已过期返回 None"""
        real_key = self._make_key(key)
        raw = await self._client.get(real_key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete(self, key: str) -> None:
        """删除键，不存在时静默忽略"""
        real_key = self._make_key(key)
        await self._client.delete(real_key)

    async def exists(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        real_key = self._make_key(key)
        return bool(await self._client.exists(real_key))

    async def keys(self, pattern: str) -> list[str]:
        """按模式匹配返回所有键名。pattern 支持 * 通配符。

        注意：生产环境高 key 数量时建议改用 SCAN 以避免阻塞。
        """
        real_pattern = self._make_key(pattern)
        raw_keys = await self._client.keys(real_pattern)
        return [self._strip_prefix(k) for k in raw_keys]

    async def count(self, pattern: str) -> int:
        """按模式匹配计数"""
        return len(await self.keys(pattern))

    async def set_nx(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set if not exists. 返回 True 如果成功设置，False 如果键已存在。"""
        real_key = self._make_key(key)
        data = json.dumps(value)
        if ttl is not None:
            result = await self._client.set(real_key, data, nx=True, ex=ttl)
        else:
            result = await self._client.set(real_key, data, nx=True)
        return result is not None

    async def close(self) -> None:
        """关闭 Redis 连接"""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """检查 Redis 是否健康可用"""
        try:
            return await self._client.ping()
        except Exception:
            return False
