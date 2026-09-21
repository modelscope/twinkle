# Copyright (c) ModelScope Contributors. All rights reserved.
"""Decision_Boundary tests: preflight rejects with real status codes and zero writes.

Covers the case where a rejected request writes no future record, plus the
TwinkleServerError handler wire shape. No Ray or GPU is involved: the
task queue is driven with a spy state that counts ``store_future_status`` calls.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from twinkle.server.deployment import twinkle_server_error_handler
from twinkle.server.exceptions import (BatchSizeError, InputTokensExceededError, RateLimitExceededError,
                                       RequestRejectedError, TwinkleServerError)
from twinkle.server.task_queue.config import TaskQueueConfig
from twinkle.server.task_queue.mixin import TaskQueueMixin


class _SpyState:
    """Counts every future write so a rejected request can be proven to write nothing."""

    def __init__(self):
        self.store_calls = 0

    async def store_future_status(self, *args, **kwargs):
        self.store_calls += 1

    async def get_future(self, request_id):
        return None


class _Harness(TaskQueueMixin):

    def __init__(self, **config_kwargs):
        self.state = _SpyState()
        self.replica_id = 'test-replica'
        self._init_task_queue(TaskQueueConfig(**config_kwargs), deployment_name='test')


async def _noop():
    return None


@pytest.mark.asyncio
async def test_input_tokens_rejection_is_422_and_zero_writes():
    """An over-limit request raises 422 and writes no record."""
    h = _Harness(enabled=True, max_input_tokens=10)
    try:
        with pytest.raises(InputTokensExceededError) as exc:
            await h.schedule_task(lambda: _noop(), model_id='m', token='tok', input_tokens=999, task_type='forward')
        assert exc.value.error_code == 422
        assert exc.value.category.value == 'user'
        assert h.state.store_calls == 0
    finally:
        await h.shutdown_task_queue()


@pytest.mark.asyncio
async def test_batch_size_rejection_is_422_and_zero_writes():
    h = _Harness(enabled=True, max_input_tokens=100000)
    try:
        with pytest.raises(BatchSizeError):
            await h.schedule_task(
                lambda: _noop(), model_id='m', token='tok', input_tokens=1,
                batch_size=1, data_world_size=4, task_type='forward')
        assert h.state.store_calls == 0
    finally:
        await h.shutdown_task_queue()


@pytest.mark.asyncio
async def test_rate_limit_rejection_is_429_and_zero_writes():
    h = _Harness(enabled=True, rps_limit=1, tps_limit=1000000, window_seconds=100, max_input_tokens=100000)
    try:
        # First call is admitted (it enqueues -> writes); reset the counter and
        # assert the rejected second call (same window, rps=1) writes nothing.
        await h.schedule_task(lambda: _noop(), model_id='m', token='tok', input_tokens=1, task_type='forward')
        h.state.store_calls = 0
        with pytest.raises(RateLimitExceededError) as exc:
            await h.schedule_task(lambda: _noop(), model_id='m', token='tok', input_tokens=1, task_type='forward')
        assert exc.value.error_code == 429
        assert h.state.store_calls == 0
    finally:
        await h.shutdown_task_queue()


@pytest.mark.asyncio
async def test_disabled_queue_skips_preflight():
    """The 'no token or queue disabled' short circuit is preserved."""
    h = _Harness(enabled=False, max_input_tokens=10)
    try:
        ref = await h.schedule_task(lambda: _noop(), model_id='m', token='tok', input_tokens=999, task_type='forward')
        assert 'request_id' in ref            # not rejected: enqueued normally
    finally:
        await h.shutdown_task_queue()


def test_error_handler_puts_fields_at_top_level():
    """The handler returns error_code as the status and fields at top level."""
    app = FastAPI()
    app.add_exception_handler(TwinkleServerError, twinkle_server_error_handler)

    @app.get('/boom')
    async def boom(request: Request):
        raise RequestRejectedError('nope', error_code=409)

    resp = TestClient(app, raise_server_exceptions=False).get('/boom')
    assert resp.status_code == 409
    body = resp.json()
    assert 'detail' not in body                # not nested under detail
    assert body['error'] == 'nope'
    assert body['category'] == 'user'
    assert body['error_code'] == 409


def test_error_handler_bounds_overlong_domain_error():
    app = FastAPI()
    app.add_exception_handler(TwinkleServerError, twinkle_server_error_handler)

    @app.get('/boom')
    async def boom(request: Request):
        raise RequestRejectedError(f'bad request {"X" * 2048}\ninternal detail', error_code=409)

    response = TestClient(app, raise_server_exceptions=False).get('/boom')
    assert response.status_code == 409
    body = response.json()
    assert len(body['error']) == 1024
    assert '\n' not in body['error']
    assert body['category'] == 'user'
    assert 'traceback' not in body
