"""Backend factory for creating StateBackend instances based on configuration."""
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from .base import StateBackend
from .memory_backend import MemoryBackend

logger = logging.getLogger(__name__)


class PersistenceConfig(BaseModel):
    """Configuration for state persistence backend."""
    mode: Literal['memory', 'file', 'redis'] = 'memory'
    file_path: str | None = None  # required for file mode
    redis_url: str | None = None  # required for redis mode
    key_prefix: str = ''  # optional global key prefix


def create_backend(config: PersistenceConfig | None = None) -> StateBackend:
    """Create a StateBackend instance based on persistence configuration.

    Args:
        config: Persistence configuration. Defaults to memory mode if None.

    Returns:
        A configured StateBackend instance.

    Raises:
        ValueError: If required config fields are missing for the selected mode.
        ImportError: If required packages are not installed.
    """
    if config is None:
        config = PersistenceConfig()

    match config.mode:
        case 'memory':
            return MemoryBackend()
        case 'file':
            if not config.file_path:
                raise ValueError('file_path is required for file persistence mode')
            from .file_backend import FileBackend
            return FileBackend(config.file_path)
        case 'redis':
            if not config.redis_url:
                raise ValueError('redis_url is required for redis persistence mode')
            from .redis_backend import RedisBackend
            return RedisBackend(config.redis_url, key_prefix=config.key_prefix)
        case _:
            raise ValueError(f'Unknown persistence mode: {config.mode}')
