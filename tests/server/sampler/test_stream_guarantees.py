from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from twinkle.server.sampler.app import SamplerManagement
from twinkle.server.sampler.twinkle_handlers import _await_generation, _register_twinkle_sampler_routes, _stream_queue


class _BlockingQueue:

    def __init__(self) -> None:
        self.released = threading.Event()
        self.get_exited = threading.Event()
        self.closed = False

    def get(self):
        self.released.wait()
        self.get_exited.set()
        return 'sentinel'

    def shutdown(self, *, force: bool) -> None:
        assert force is True
        self.closed = True
        self.released.set()


@pytest.mark.asyncio
async def test_sampler_request_refreshes_replica_liveness():
    service = SamplerManagement.__new__(SamplerManagement)
    service.replica_id = 'sampler-replica'
    service.state = SimpleNamespace(touch_replica_last_seen=AsyncMock())
    service._ensure_sticky = AsyncMock()
    service._ensure_state_cleanup_started = AsyncMock()
    request = SimpleNamespace(
        headers={'Authorization': 'Bearer token'}, state=SimpleNamespace(token='token'))

    assert await service._on_request_start(request) == 'token'
    service.state.touch_replica_last_seen.assert_awaited_once_with('sampler-replica')


@pytest.mark.asyncio
async def test_stream_without_actor_returns_structured_error():
    service = SimpleNamespace(
        sampler=SimpleNamespace(_actors=[]),
        _on_request_start=AsyncMock(return_value='token'),
    )
    app = FastAPI()
    _register_twinkle_sampler_routes(app, lambda: service)
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/twinkle/sample_stream')
    request = SimpleNamespace(state=SimpleNamespace(request_id='request'))
    body = SimpleNamespace(adapter_name='', adapter_uri=None, inputs={'input_ids': [1]}, sampling_params=None)

    response = await route.endpoint(request, body, service)
    chunks = [chunk async for chunk in response.body_iterator]
    payload = json.loads(chunks[0])
    assert payload['category'] == 'server'
    assert payload['error_code'] == 503
    assert payload['request_id'].startswith('req_')


@pytest.mark.asyncio
async def test_stream_timeout_returns_error_payload_and_closes_queue():
    queue = _BlockingQueue()
    chunks = [
        chunk async for chunk in _stream_queue(
            queue,
            sentinel='sentinel',
            request_id='req-stream',
            total_timeout=0.05,
            single_get_timeout=0.05,
        )
    ]

    payload = json.loads(chunks[0])
    assert payload['category'] == 'server'
    assert payload['error_code'] == 504
    assert payload['request_id'] == 'req-stream'
    assert queue.closed is True
    assert queue.get_exited.wait(timeout=5)


class _GenerationService:

    def __init__(self) -> None:
        self.cancelled = False
        self.sampler = SimpleNamespace(
            get_generation_status=lambda _submission_id: {'status': 'running'},
            collect_generation=lambda _submission_id: [],
            cancel_generation=self._cancel,
        )

    async def call_backend(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    def _cancel(self, _submission_id: str) -> None:
        self.cancelled = True


@pytest.mark.asyncio
async def test_generation_poll_has_total_timeout_and_cancels():
    service = _GenerationService()

    with pytest.raises(asyncio.TimeoutError):
        await _await_generation(service, 'submission', timeout=0.05)

    assert service.cancelled is True
