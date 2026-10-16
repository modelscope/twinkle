# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
import uuid

import pytest
import pytest_asyncio

from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.state import ServerState
from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.state.backend.redis_backend import RedisBackend


@pytest_asyncio.fixture(params=['memory', 'redis'])
async def state(request):
    if request.param == 'redis':
        url = os.environ.get('TWINKLE_TEST_REDIS_URL')
        if not url:
            pytest.skip('TWINKLE_TEST_REDIS_URL is required for the Redis regression')
        backend = RedisBackend(url, key_prefix='sequence-test-' + uuid.uuid4().hex + ':')
    else:
        backend = RayActorBackend()
    yield ServerState(backend=backend)
    if request.param == 'redis':
        for key in await backend.keys('*'):
            await backend.delete(key)
        await backend.close()


@pytest.mark.asyncio
async def test_expired_retry_receipts_cannot_execute_old_sequences_again(state, monkeypatch):
    backend = state.backend
    now = time.time()
    monkeypatch.setattr('twinkle.server.state.server_state.time.time', lambda: now)
    assert await state.claim_training_seq('training', 'native::session::step', 1, 'first', 10) is None
    assert await state.claim_training_seq('training', 'native::session::step', 1, 'retry', 10) == 'first'
    now += 11
    assert await state.claim_training_seq('training', 'native::session::step', 2, 'second', 10) is None
    with pytest.raises(RequestRejectedError, match='retry window expired'):
        await state.claim_training_seq('training', 'native::session::step', 1, 'late', 10)
    ledger = await backend.get('seq_ledger::training::native::session::step')
    assert list(ledger['receipts']) == ['2']
    assert len(await backend.keys('seq_ledger::*')) == 1


@pytest.mark.asyncio
async def test_out_of_order_sequences_and_preflight_release_remain_retryable(state):
    assert await state.claim_training_seq('training', 'tinker::step', 2, 'second', 10) is None
    assert await state.claim_training_seq('training', 'tinker::step', 1, 'first', 10) is None
    await state.release_training_seq('training', 'tinker::step', 1, 'another-request')
    assert await state.claim_training_seq('training', 'tinker::step', 1, 'retry', 10) == 'first'
    await state.release_training_seq('training', 'tinker::step', 1, 'first')
    assert await state.claim_training_seq('training', 'tinker::step', 1, 'retry', 10) is None
    assert await state.claim_training_seq('other-training', 'tinker::step', 1, 'other', 10) is None


@pytest.mark.asyncio
async def test_retiring_a_handle_removes_native_and_tinker_stream_ledgers(state):
    from tests.server.training.test_controller import create, setup
    controller, state, _, _ = await setup(backend=state.backend)
    try:
        identity = await create(controller, state, 'dedup-retirement')
        for stream in ('native::session::step', 'tinker::optim_step'):
            await state.claim_training_seq(identity, stream, 1, stream, 10)
        assert len(await state.backend.keys(f'seq_ledger::{identity}::*')) == 2
        await controller.release(identity, 'owner', 'session')
        assert await state.backend.keys(f'seq_ledger::{identity}::*') == []
    finally:
        await controller.shutdown()
