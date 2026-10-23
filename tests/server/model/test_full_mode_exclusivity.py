# Copyright (c) ModelScope Contributors. All rights reserved.
"""Full-mode exclusivity must be one atomic state transition, not check-then-register."""
from __future__ import annotations

import asyncio

import pytest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from starlette.requests import Request

from tests.server.state.fake_backend import FakeBackend
from tests.server.training.helpers import request, setup
from twinkle.server.config.training import TrainingConfig
from twinkle.server.model.pool.router import TrainingRouter


async def _setup_full_instance():
    """Full-mode slots belong to the controller, not the retired model create route."""
    controller, state, runtime, _ = await setup(backend=FakeBackend())
    raw = controller.config.model_dump()
    raw['models'][1]['config']['max_replicas'] = 1
    controller.config = TrainingConfig.model_validate(raw)
    runtime.start_gate.clear()
    return controller, state, runtime


@pytest.mark.asyncio
async def test_concurrent_full_mode_claims_have_exactly_one_winner():
    controller, state, runtime = await _setup_full_instance()
    try:
        refs = await asyncio.gather(*(controller.create(request(name, mode='full'))
                                      for name in ('tenant-a', 'tenant-b')))
        assert len(controller.instances) == 1
        assert len(next(iter(controller.instances.values())).training_ids) == 1
        runtime.start_gate.set()
        await controller.wait_for_jobs()
        records = [await state.get_future(ref['request_id']) for ref in refs]
        assert sum(record['status'] == 'completed' for record in records) == 1
        assert sum(record['status'] == 'failed' and record['failure']['reason_code'] == 'capacity_unavailable'
                   for record in records) == 1
        assert await state._backend.get('token_count::owner') == 1
    finally:
        runtime.start_gate.set()
        await controller.wait_for_jobs()
        await controller.shutdown()


@pytest.mark.asyncio
async def test_same_tenant_may_reclaim_its_own_resource():
    controller, _, runtime = await _setup_full_instance()
    try:
        refs = await asyncio.gather(*(controller.create(request('same', mode='full')) for _ in range(2)))
        assert refs[0] == refs[1]
        assert len(controller.instances) == 1
        assert len(next(iter(controller.instances.values())).training_ids) == 1
    finally:
        runtime.start_gate.set()
        await controller.wait_for_jobs()
        await controller.shutdown()


def _request(body):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
    incoming = Request({'type': 'http', 'method': 'POST', 'path': '/twinkle/add_adapter_to_model', 'headers': []}, receive)
    incoming.state.session_id = 'session'
    incoming.state.token = 'owner'
    incoming.state.request_id = 'req'
    return incoming


@pytest.mark.asyncio
async def test_concurrent_handler_creates_have_exactly_one_winner():
    """Drive the current catalog handler into real controller reservations."""
    controller, state, runtime = await _setup_full_instance()
    router = TrainingRouter.__new__(TrainingRouter)
    router.state = state
    router.config = controller.config
    router.model = SimpleNamespace(name='small-full')
    router.controller = SimpleNamespace(create=SimpleNamespace(remote=AsyncMock(side_effect=controller.create)))

    async def create_remote(payload):
        from twinkle.server.state.training_records import CreateTraining
        return await controller.create(CreateTraining.model_validate(payload))
    router.controller.create.remote.side_effect = create_remote
    try:
        refs = await asyncio.gather(*(router.native_request(_request({
            'client_create_id': name, 'adapter_name': name,
        }), 'add_adapter_to_model') for name in ('adapter-a', 'adapter-b')))
        assert len(controller.instances) == 1
        assert len(next(iter(controller.instances.values())).training_ids) == 1
        runtime.start_gate.set()
        await controller.wait_for_jobs()
        records = [await state.get_future(ref.request_id) for ref in refs]
        assert sum(record['status'] == 'completed' for record in records) == 1
        assert sum(record['status'] == 'failed' and record['failure']['reason_code'] == 'capacity_unavailable'
                   for record in records) == 1
        assert await state._backend.get('token_count::owner') == 1
    finally:
        runtime.start_gate.set()
        await controller.wait_for_jobs()
        await controller.shutdown()
