# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persistence backend configuration model."""
from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict
from typing import Literal

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

    model_config = ConfigDict(extra='forbid')

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
