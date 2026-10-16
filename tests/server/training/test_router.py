# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from tests.server.state.fake_backend import FakeBackend
from tests.server.training.test_runtime_queue import settle
from twinkle.hub import HubOperation
from twinkle.server.state import ServerState
from twinkle.server.training.router import TrainingRouter
from twinkle.server.training.store import TrainingStore
from twinkle.server.training.catalog import TrainingCatalog
from twinkle.server.config.training import TrainingConfig


def http_request(body):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
    request = Request({'type': 'http', 'method': 'POST', 'path': '/upload_to_hub', 'headers': []}, receive)
    request.state.token = 'owner'
    request.state.session_id = ''
    return request


@pytest.mark.asyncio
async def test_upload_after_recycle_is_cpu_only_and_preserves_future_ownership(monkeypatch, tmp_path):
    router = TrainingRouter.__new__(TrainingRouter)
    router.state = ServerState(backend=FakeBackend())
    router.store = TrainingStore(router.state, 'upload-test')
    router.replica_id = 'router'
    await router.state.register_replica(router.replica_id, 0)
    await router.state.touch_replica_last_seen(router.replica_id)
    router._init_task_queue()
    router.controller = Mock(side_effect=AssertionError('Upload must not create or route a training'))
    upload = Mock(return_value=None)
    monkeypatch.setattr(HubOperation, 'push_to_hub', upload)
    try:
        response = await router.native_request(http_request({
            'checkpoint_dir': str(tmp_path), 'hub_model_id': 'owner/saved', 'hub_token': 'hub-token',
        }), 'upload_to_hub')
        future = await settle(router.state, response.request_id)
        assert future['status'] == 'completed'
        upload.assert_called_once_with(repo_id='owner/saved', folder_path=str(tmp_path), token='hub-token', private=True)
        assert await router.state._backend.get('future_owner::' + response.request_id) == 'owner'
        assert not await router.state._backend.keys('model::*')
    finally:
        await router.shutdown_task_queue()


@pytest.mark.asyncio
@pytest.mark.parametrize('rank', [8, 0, 33, True])
async def test_native_lora_admission_does_not_import_peft(monkeypatch, rank):
    import builtins
    from unittest.mock import AsyncMock
    from twinkle.server.exceptions import RequestRejectedError
    router = TrainingRouter.__new__(TrainingRouter)
    router.model = SimpleNamespace(name='small')
    router.state = ServerState(backend=FakeBackend())
    await router.state.create_session({'session_id': 'session'})
    router.catalog = TrainingCatalog(TrainingConfig(models=[{
        'name': 'small', 'model_id': 'weights', 'backend': 'transformers', 'device_type': 'cuda',
        'modes': {'lora': {'num_devices': 1, 'device_mesh': {'dp_size': 1}}},
    }]))
    async def create(payload):
        await router.state.store_future_status('created', 'pending', None)
        return {'request_id': 'created'}
    router.controller = SimpleNamespace(create=SimpleNamespace(remote=AsyncMock(side_effect=create)))
    original_import = builtins.__import__
    def cpu_import(name, *args, **kwargs):
        if name == 'peft' or name.startswith('peft.'):
            raise AssertionError('CPU admission must not import PEFT')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', cpu_import)
    request = http_request({'client_create_id': 'create', 'adapter_name': 'same',
                            'config': json.dumps({'_TWINKLE_TYPE_': 'LoraConfig', 'r': rank})})
    request.state.session_id = 'session'
    if type(rank) is int and rank == 8:
        result = await router.native_request(request, 'add_adapter_to_model')
        assert result.request_id == 'created'
        router.controller.create.remote.assert_awaited_once()
    else:
        with pytest.raises(RequestRejectedError, match='rank'):
            await router.native_request(request, 'add_adapter_to_model')
        router.controller.create.remote.assert_not_awaited()
