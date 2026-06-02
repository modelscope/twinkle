"""Tests for RedisBackend - using mocks since no real Redis is available."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Skip entire module if redis package not available
redis = pytest.importorskip('redis')

from twinkle.server.state.backend.redis_backend import RedisBackend  # noqa: E402


@pytest.fixture
def mock_redis_client():
    """Create a mock redis.asyncio client."""
    client = AsyncMock()
    client.set = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=1)
    client.keys = AsyncMock(return_value=[])
    client.ping = AsyncMock(return_value=True)
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def backend_no_prefix(mock_redis_client):
    """RedisBackend with no key prefix."""
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        backend = RedisBackend('redis://localhost:6379')
    return backend


@pytest.fixture
def backend_with_prefix(mock_redis_client):
    """RedisBackend with key prefix."""
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        backend = RedisBackend('redis://localhost:6379', key_prefix='twinkle:')
    return backend


# ---- SET ----


@pytest.mark.asyncio
async def test_set_without_ttl(backend_no_prefix, mock_redis_client):
    await backend_no_prefix.set('mykey', {'data': 123})
    mock_redis_client.set.assert_called_once_with('mykey', json.dumps({'data': 123}))


@pytest.mark.asyncio
async def test_set_with_ttl(backend_no_prefix, mock_redis_client):
    await backend_no_prefix.set('mykey', 'value', ttl=60)
    mock_redis_client.set.assert_called_once_with('mykey', json.dumps('value'), ex=60)


@pytest.mark.asyncio
async def test_set_with_prefix(backend_with_prefix, mock_redis_client):
    await backend_with_prefix.set('mykey', 'val')
    mock_redis_client.set.assert_called_once_with('twinkle:mykey', json.dumps('val'))


# ---- GET ----


@pytest.mark.asyncio
async def test_get_existing_key(backend_no_prefix, mock_redis_client):
    mock_redis_client.get.return_value = json.dumps({'hello': 'world'})
    result = await backend_no_prefix.get('mykey')
    mock_redis_client.get.assert_called_once_with('mykey')
    assert result == {'hello': 'world'}


@pytest.mark.asyncio
async def test_get_nonexistent_key(backend_no_prefix, mock_redis_client):
    mock_redis_client.get.return_value = None
    result = await backend_no_prefix.get('missing')
    assert result is None


@pytest.mark.asyncio
async def test_get_with_prefix(backend_with_prefix, mock_redis_client):
    mock_redis_client.get.return_value = json.dumps('data')
    result = await backend_with_prefix.get('mykey')
    mock_redis_client.get.assert_called_once_with('twinkle:mykey')
    assert result == 'data'


# ---- DELETE ----


@pytest.mark.asyncio
async def test_delete(backend_no_prefix, mock_redis_client):
    await backend_no_prefix.delete('mykey')
    mock_redis_client.delete.assert_called_once_with('mykey')


@pytest.mark.asyncio
async def test_delete_with_prefix(backend_with_prefix, mock_redis_client):
    await backend_with_prefix.delete('mykey')
    mock_redis_client.delete.assert_called_once_with('twinkle:mykey')


# ---- EXISTS ----


@pytest.mark.asyncio
async def test_exists_true(backend_no_prefix, mock_redis_client):
    mock_redis_client.exists.return_value = 1
    result = await backend_no_prefix.exists('mykey')
    assert result is True
    mock_redis_client.exists.assert_called_once_with('mykey')


@pytest.mark.asyncio
async def test_exists_false(backend_no_prefix, mock_redis_client):
    mock_redis_client.exists.return_value = 0
    result = await backend_no_prefix.exists('mykey')
    assert result is False


# ---- KEYS ----


@pytest.mark.asyncio
async def test_keys_pattern(backend_no_prefix, mock_redis_client):
    mock_redis_client.keys.return_value = ['session::a', 'session::b']
    result = await backend_no_prefix.keys('session::*')
    mock_redis_client.keys.assert_called_once_with('session::*')
    assert result == ['session::a', 'session::b']


@pytest.mark.asyncio
async def test_keys_with_prefix(backend_with_prefix, mock_redis_client):
    mock_redis_client.keys.return_value = ['twinkle:session::a', 'twinkle:session::b']
    result = await backend_with_prefix.keys('session::*')
    mock_redis_client.keys.assert_called_once_with('twinkle:session::*')
    # Result should have prefix stripped
    assert result == ['session::a', 'session::b']


# ---- COUNT ----


@pytest.mark.asyncio
async def test_count(backend_no_prefix, mock_redis_client):
    mock_redis_client.keys.return_value = ['a', 'b', 'c']
    result = await backend_no_prefix.count('*')
    assert result == 3


# ---- SET_NX ----


@pytest.mark.asyncio
async def test_set_nx_success(backend_no_prefix, mock_redis_client):
    mock_redis_client.set.return_value = True  # nx succeeded
    result = await backend_no_prefix.set_nx('newkey', {'value': 1})
    mock_redis_client.set.assert_called_once_with('newkey', json.dumps({'value': 1}), nx=True)
    assert result is True


@pytest.mark.asyncio
async def test_set_nx_failure(backend_no_prefix, mock_redis_client):
    mock_redis_client.set.return_value = None  # nx failed (key exists)
    result = await backend_no_prefix.set_nx('existing', 'val')
    assert result is False


# ---- HEALTH CHECK ----


@pytest.mark.asyncio
async def test_health_check_healthy(backend_no_prefix, mock_redis_client):
    mock_redis_client.ping.return_value = True
    result = await backend_no_prefix.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_unhealthy(backend_no_prefix, mock_redis_client):
    mock_redis_client.ping.side_effect = ConnectionError('offline')
    result = await backend_no_prefix.health_check()
    assert result is False


# ---- CLOSE ----


@pytest.mark.asyncio
async def test_close(backend_no_prefix, mock_redis_client):
    await backend_no_prefix.close()
    mock_redis_client.aclose.assert_called_once()


# ---- JSON Serialization ----


@pytest.mark.asyncio
async def test_json_serialization_complex_types(backend_no_prefix, mock_redis_client):
    """Values should be JSON-serialized before storage."""
    complex_value = {'list': [1, 2, 3], 'nested': {'key': 'val'}, 'null': None}
    await backend_no_prefix.set('complex', complex_value)
    expected_json = json.dumps(complex_value)
    mock_redis_client.set.assert_called_once_with('complex', expected_json)
