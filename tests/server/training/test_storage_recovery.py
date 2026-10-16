# Copyright (c) ModelScope Contributors. All rights reserved.
"""The lifecycle uses the same recovery semantics on real memory and Redis backends."""
import os
import uuid

import pytest

from twinkle.server.state.backend.memory_backend import RayActorBackend
from twinkle.server.state.backend.redis_backend import RedisBackend
from twinkle.server.training.controller import TrainingController
from twinkle.server.training.store import TrainingStore
from tests.server.training.test_controller import create, request, setup


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['memory', 'redis'])
async def test_pending_operation_recovery_preserves_creation_identity_and_quota(mode):
    if mode == 'redis' and not os.environ.get('TWINKLE_TEST_REDIS_URL'):
        pytest.skip('Set TWINKLE_TEST_REDIS_URL to a dedicated local Redis')
    backend = (RayActorBackend() if mode == 'memory' else
               RedisBackend(os.environ['TWINKLE_TEST_REDIS_URL'], key_prefix='training-test-' + uuid.uuid4().hex + ':'))
    try:
        controller, state, runtime, clock = await setup(backend=backend)
        identity = await create(controller, state, 'retry')
        original = await controller.create(request('retry'))
        await controller.admit(identity, 'owner', 'session', 'queued')
        await controller.store.future('queued', 'queued', identity)
        replacement = TrainingController(controller.config, TrainingStore(state, 'unit'), runtime, clock=clock)
        await replacement.initialize()
        assert (await state.get_future('queued'))['failure']['reason_code'] == 'orphaned_replica'
        assert await replacement.create(request('retry')) == original
        assert not runtime.allocations
        assert await state.get_model_metadata(identity) is None
        assert (await backend.get('token_count::owner') or 0) == 0
        new = await create(replacement, state, 'new')
        assert new != identity
        assert await backend.get('token_count::owner') == 1
        await replacement.shutdown()
    finally:
        if mode == 'redis':
            for key in await backend.keys('*'):
                await backend.delete(key)
            await backend.close()


@pytest.mark.asyncio
async def test_managed_memory_actor_restart_reconnects_existing_handles():
    import asyncio
    import ray
    backend = RayActorBackend(namespace='memory-recovery-' + uuid.uuid4().hex)
    try:
        await backend.set('before', 'old')
        ray.kill(backend._actor, no_restart=False)
        for _ in range(100):
            try:
                if await backend.health_check() and await backend.get('before') is None:break
            except ray.exceptions.RayError:pass
            await asyncio.sleep(0.1)
        else:raise AssertionError('State actor did not restart')
        assert await backend.get('before') is None
        await backend.set('after', 'new')
        assert await backend.get('after') == 'new'
    finally:
        ray.kill(backend._actor, no_restart=True)
