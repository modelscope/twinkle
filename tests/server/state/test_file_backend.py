"""Tests for FileBackend - JSON file-based state backend."""
from __future__ import annotations

import asyncio
import json
import os
import pytest
import tempfile
import time

from twinkle.server.state.backend.file_backend import FileBackend


@pytest.fixture
def tmp_file():
    """Provide a temporary file path, deleted before use so FileBackend creates fresh."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# ---- Basic CRUD ----


@pytest.mark.asyncio
async def test_set_and_get(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('key1', {'hello': 'world'})
    result = await backend.get('key1')
    assert result == {'hello': 'world'}


@pytest.mark.asyncio
async def test_get_nonexistent_key(tmp_file):
    backend = FileBackend(tmp_file)
    result = await backend.get('nonexistent')
    assert result is None


@pytest.mark.asyncio
async def test_delete(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('key1', 'value1')
    await backend.delete('key1')
    result = await backend.get('key1')
    assert result is None


@pytest.mark.asyncio
async def test_delete_nonexistent_key(tmp_file):
    """Deleting a key that doesn't exist should not raise."""
    backend = FileBackend(tmp_file)
    await backend.delete('nonexistent')  # Should not raise


@pytest.mark.asyncio
async def test_exists(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('key1', 'value1')
    assert await backend.exists('key1') is True
    assert await backend.exists('key2') is False


# ---- TTL Expiry ----


@pytest.mark.asyncio
async def test_ttl_expiry(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('ephemeral', 'data', ttl=1)
    # Immediately should exist
    assert await backend.get('ephemeral') == 'data'
    assert await backend.exists('ephemeral') is True
    # Wait for expiry
    time.sleep(1.1)
    assert await backend.get('ephemeral') is None
    assert await backend.exists('ephemeral') is False


@pytest.mark.asyncio
async def test_ttl_none_means_no_expiry(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('permanent', 'data', ttl=None)
    time.sleep(0.1)
    assert await backend.get('permanent') == 'data'


# ---- Keys Pattern Matching ----


@pytest.mark.asyncio
async def test_keys_wildcard(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('session::abc', 's1')
    await backend.set('session::def', 's2')
    await backend.set('model::xyz', 'm1')

    session_keys = await backend.keys('session::*')
    assert sorted(session_keys) == ['session::abc', 'session::def']

    model_keys = await backend.keys('model::*')
    assert model_keys == ['model::xyz']

    all_keys = await backend.keys('*')
    assert len(all_keys) == 3


@pytest.mark.asyncio
async def test_keys_excludes_expired(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('alive', 'yes')
    await backend.set('dying', 'soon', ttl=1)
    time.sleep(1.1)
    keys = await backend.keys('*')
    assert keys == ['alive']


# ---- Count ----


@pytest.mark.asyncio
async def test_count(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('a::1', 'v')
    await backend.set('a::2', 'v')
    await backend.set('b::1', 'v')
    assert await backend.count('a::*') == 2
    assert await backend.count('b::*') == 1
    assert await backend.count('*') == 3


# ---- set_nx ----


@pytest.mark.asyncio
async def test_set_nx_new_key(tmp_file):
    backend = FileBackend(tmp_file)
    result = await backend.set_nx('new_key', 'value')
    assert result is True
    assert await backend.get('new_key') == 'value'


@pytest.mark.asyncio
async def test_set_nx_existing_key(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('existing', 'original')
    result = await backend.set_nx('existing', 'new_value')
    assert result is False
    # Value should not change
    assert await backend.get('existing') == 'original'


@pytest.mark.asyncio
async def test_set_nx_expired_key(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('expired_key', 'old', ttl=1)
    time.sleep(1.1)
    # Key is expired, set_nx should succeed
    result = await backend.set_nx('expired_key', 'new_value')
    assert result is True
    assert await backend.get('expired_key') == 'new_value'


# ---- Health Check ----


@pytest.mark.asyncio
async def test_health_check(tmp_file):
    backend = FileBackend(tmp_file)
    assert await backend.health_check() is True


# ---- Auto-create File ----


@pytest.mark.asyncio
async def test_auto_create_file():
    """FileBackend should create the file if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'subdir', 'state.json')
        FileBackend(path)
        assert os.path.exists(path)
        # File should be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data == {}


# ---- Atomic Write Integrity ----


@pytest.mark.asyncio
async def test_atomic_write_integrity(tmp_file):
    """After write, reading from file should give consistent data."""
    backend = FileBackend(tmp_file)
    await backend.set('k1', {'nested': [1, 2, 3]})
    await backend.set('k2', 'simple_string')

    # Read raw file to verify structure
    with open(tmp_file, encoding='utf-8') as f:
        raw = json.load(f)
    assert 'k1' in raw
    assert raw['k1']['value'] == {'nested': [1, 2, 3]}
    assert 'k2' in raw
    assert raw['k2']['value'] == 'simple_string'


# ---- Overwrite Value ----


@pytest.mark.asyncio
async def test_overwrite_value(tmp_file):
    backend = FileBackend(tmp_file)
    await backend.set('key', 'v1')
    await backend.set('key', 'v2')
    assert await backend.get('key') == 'v2'
