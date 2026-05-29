from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StateBackend(ABC):
    """状态存储后端的统一接口。

    所有状态管理操作通过此接口进行，支持多种后端实现（内存、文件、Redis）。
    """

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """存储键值对，可选 TTL（秒）"""
        ...

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """获取值，不存在或已过期返回 None"""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除键，不存在时静默忽略"""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        ...

    @abstractmethod
    async def keys(self, pattern: str) -> list[str]:
        """按模式匹配返回所有键名。pattern 支持 * 通配符（如 'session::*'）"""
        ...

    @abstractmethod
    async def count(self, pattern: str) -> int:
        """按模式匹配计数"""
        ...

    @abstractmethod
    async def set_nx(self, key: str, value: Any) -> bool:
        """Set if not exists. 返回 True 如果成功设置，False 如果键已存在"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """关闭后端连接/释放资源"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """检查后端是否健康可用"""
        ...
