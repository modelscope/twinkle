# Copyright (c) ModelScope Contributors. All rights reserved.
"""The runtime event loop remains responsive during the first dynamic config import."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.server.state.fake_backend import FakeBackend
from tests.server.training.helpers import create, setup
from twinkle.server.model.runtime import TrainingRuntime
from twinkle.server.state.training_records import CreateTraining


@pytest.mark.asyncio
async def test_configuration_deserialization_does_not_block_event_loop(monkeypatch):
    controller, state, _, _ = await setup(backend=FakeBackend())
    identity = await create(controller, state, 'configuration-import', mode='full')
    training = controller.trainings[identity]
    entered, resume, finished = threading.Event(), threading.Event(), threading.Event()

    def deserialize(config):
        entered.set()
        resume.wait(3)
        finished.set()
        return None

    monkeypatch.setattr('twinkle.server.model.runtime.deserialize_object', deserialize)
    manager = SimpleNamespace(save=Mock())
    monkeypatch.setattr('twinkle.server.model.runtime.create_training_run_manager', lambda *args, **kw: manager)

    async def serial(factory, task_type):
        return await factory()

    runtime = SimpleNamespace(_run_lifecycle_operation=serial, resolve_model_adapter_name=lambda _: '',
                              _training_states={}, is_full_mode=True)
    request = CreateTraining(base_model='small', mode='full', token='owner', session_id='session', create_key='key',
                             client_type='twinkle', payload={'adapter_name': 'same', 'config': None})
    initializing = asyncio.create_task(TrainingRuntime.initialize_training(
        runtime, training.model_dump(), request.model_dump()))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert not finished.is_set(), 'Synchronous import blocked the event loop'
    finally:
        resume.set()
        await initializing
        await controller.shutdown()
    manager.save.assert_called_once()
