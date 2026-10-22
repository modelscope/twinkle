"""Backend factory for creating StateBackend instances based on configuration."""
from __future__ import annotations

from twinkle.server.config.persistence import PersistenceConfig
from twinkle.utils import get_logger
from .base import StateBackend

logger = get_logger()


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
            # Deferred import keeps the module-level import graph light; the Twinkle
            # server always runs on Ray Serve, so ``ray`` is available here.
            from .memory_backend import RayActorBackend
            return RayActorBackend(key_prefix=config.key_prefix)
        case 'redis':
            if not config.redis_url:
                raise ValueError('redis_url is required for redis persistence mode')
            from .redis_backend import RedisBackend
            return RedisBackend(config.redis_url, key_prefix=config.key_prefix)
        case _:
            raise ValueError(f'Unknown persistence mode: {config.mode}')
