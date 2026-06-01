"""Backend factory for creating StateBackend instances based on configuration."""
from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel

from .base import StateBackend
from .memory_backend import MemoryBackend

logger = logging.getLogger(__name__)


# Env var keys propagated by the launcher so that any Ray worker can rebuild
# the same PersistenceConfig regardless of which deployment initializes the
# ServerState actor first.
PERSISTENCE_ENV_KEYS: tuple[str, ...] = (
    'TWINKLE_PERSISTENCE_MODE',
    'TWINKLE_PERSISTENCE_FILE_PATH',
    'TWINKLE_PERSISTENCE_REDIS_URL',
    'TWINKLE_PERSISTENCE_KEY_PREFIX',
)


class PersistenceConfig(BaseModel):
    """Configuration for state persistence backend."""
    mode: Literal['memory', 'file', 'redis'] = 'memory'
    file_path: str | None = None  # required for file mode
    redis_url: str | None = None  # required for redis mode
    key_prefix: str = ''  # optional global key prefix

    def to_env_vars(self) -> dict[str, str]:
        """Serialize this config to env var key/value pairs for worker propagation."""
        env: dict[str, str] = {'TWINKLE_PERSISTENCE_MODE': self.mode}
        if self.file_path:
            env['TWINKLE_PERSISTENCE_FILE_PATH'] = self.file_path
        if self.redis_url:
            env['TWINKLE_PERSISTENCE_REDIS_URL'] = self.redis_url
        if self.key_prefix:
            env['TWINKLE_PERSISTENCE_KEY_PREFIX'] = self.key_prefix
        return env

    @classmethod
    def from_env(cls) -> PersistenceConfig | None:
        """Reconstruct a PersistenceConfig from env vars set by the launcher.

        Returns ``None`` when ``TWINKLE_PERSISTENCE_MODE`` is unset, so callers
        can distinguish "no env-configured persistence" from "memory mode".
        """
        mode = os.environ.get('TWINKLE_PERSISTENCE_MODE')
        if not mode:
            return None
        return cls(
            mode=mode,
            file_path=os.environ.get('TWINKLE_PERSISTENCE_FILE_PATH'),
            redis_url=os.environ.get('TWINKLE_PERSISTENCE_REDIS_URL'),
            key_prefix=os.environ.get('TWINKLE_PERSISTENCE_KEY_PREFIX', ''),
        )


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
