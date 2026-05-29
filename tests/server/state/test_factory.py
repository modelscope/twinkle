"""Tests for backend factory - create_backend function."""
from __future__ import annotations

import os
import tempfile

import pytest

from twinkle.server.state.backend.factory import PersistenceConfig, create_backend
from twinkle.server.state.backend.file_backend import FileBackend
from twinkle.server.state.backend.memory_backend import MemoryBackend


# ---- Memory Mode ----

def test_create_backend_none_returns_memory():
    """Passing None should return MemoryBackend (default mode)."""
    backend = create_backend(None)
    assert isinstance(backend, MemoryBackend)


def test_create_backend_memory_mode():
    """Explicit memory mode should return MemoryBackend."""
    config = PersistenceConfig(mode="memory")
    backend = create_backend(config)
    assert isinstance(backend, MemoryBackend)


# ---- File Mode ----

def test_create_backend_file_mode():
    """File mode with file_path should return FileBackend."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    try:
        os.unlink(path)  # Let FileBackend create it
        config = PersistenceConfig(mode="file", file_path=path)
        backend = create_backend(config)
        assert isinstance(backend, FileBackend)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_create_backend_file_mode_missing_path():
    """File mode without file_path should raise ValueError."""
    config = PersistenceConfig(mode="file")
    with pytest.raises(ValueError, match="file_path"):
        create_backend(config)


# ---- Redis Mode ----

def test_create_backend_redis_mode():
    """Redis mode with redis_url should return RedisBackend (if redis available)."""
    try:
        import redis  # noqa: F401
    except ImportError:
        pytest.skip("redis package not available")

    from unittest.mock import patch, MagicMock
    from twinkle.server.state.backend.redis_backend import RedisBackend

    with patch("redis.asyncio.from_url", return_value=MagicMock()):
        config = PersistenceConfig(mode="redis", redis_url="redis://localhost:6379")
        backend = create_backend(config)
        assert isinstance(backend, RedisBackend)


def test_create_backend_redis_mode_missing_url():
    """Redis mode without redis_url should raise ValueError."""
    config = PersistenceConfig(mode="redis")
    with pytest.raises(ValueError, match="redis_url"):
        create_backend(config)


# ---- PersistenceConfig Defaults ----

def test_persistence_config_defaults():
    """PersistenceConfig should have sensible defaults."""
    config = PersistenceConfig()
    assert config.mode == "memory"
    assert config.file_path is None
    assert config.redis_url is None
    assert config.key_prefix == ""
