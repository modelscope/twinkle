# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import json
import threading
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from tests.server.state.fake_backend import FakeBackend
from tests.server.training.helpers import settle
from twinkle.hub import HubOperation
from twinkle.server.checkpoint import create_checkpoint_manager
from twinkle.server.gateway.use_cases import poll_future
from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError
from twinkle.server.state import ServerState
from twinkle.server.model.pool.router import TrainingRouter
from twinkle.server.config.training import TrainingConfig


def http_request(body, token='owner'):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}
    request = Request({'type': 'http', 'method': 'POST', 'path': '/upload_to_hub', 'headers': []}, receive)
    request.state.token = token
    request.state.session_id = ''
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [True, False], ids=['owned', 'outside-tenant-directory'])
async def test_upload_after_recycle_is_cpu_only_and_preserves_future_ownership(monkeypatch, tmp_path, owned):
    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    manager = create_checkpoint_manager('owner', client_type='twinkle')
    checkpoint = (manager.training_run_manager.get_base_dir() if owned else tmp_path) / 'checkpoint'
    checkpoint.mkdir(parents=True)
    router = TrainingRouter.__new__(TrainingRouter)
    router.state = ServerState(backend=FakeBackend())
    router.store = router.state.create_training_store('upload-test')
    router.replica_id = 'router'
    await router.state.register_replica(router.replica_id, 0)
    await router.state.touch_replica_last_seen(router.replica_id)
    router._init_task_queue()
    router.controller = Mock(side_effect=AssertionError('Upload must not create or route a training'))
    finish = threading.Event()

    def transfer(*, repo_id, folder_path, token, report):
        report(stage='uploading', uploaded_bytes=5, total_bytes=10, committed_bytes=0)
        assert finish.wait(5), 'poll did not receive progress during upload'
        report(stage='batch_committed', uploaded_bytes=10, total_bytes=10, committed_bytes=10)
        return 'commit-sha'

    upload = Mock(side_effect=transfer)
    monkeypatch.setattr(HubOperation, 'upload_checkpoint', upload)
    try:
        response = await router.native_request(http_request({
            'checkpoint_dir': str(checkpoint), 'hub_model_id': 'owner/saved', 'hub_token': 'hub-token',
        }), 'upload_to_hub')
        if owned:
            with pytest.raises(ResourceNotFoundError):
                await router.state.assert_future_owner(response.request_id, 'another-tenant')
            outcome = await asyncio.wait_for(poll_future(router.state, response.request_id, after_progress=1), 3)
            assert outcome.record['status'] == 'running'
            assert outcome.record['progress']['uploaded_bytes'] == 5
            finish.set()
            future = await settle(router.state, response.request_id)
            assert future['status'] == 'completed'
            assert future['result'] == {'repo_id': 'owner/saved', 'revision': 'commit-sha'}
            await router.state.report_future_progress(response.request_id, {'stage': 'batch_failed'})
            assert (await router.state.get_future(response.request_id))['progress']['stage'] == 'batch_committed'
            assert upload.call_count == 1
            assert upload.call_args.kwargs['folder_path'] == str(checkpoint)
            assert upload.call_args.kwargs['token'] == 'hub-token'
        else:
            future = await settle(router.state, response.request_id)
            assert future['status'] == 'failed'
            upload.assert_not_called()
        assert (await router.state.get_future(response.request_id))['owner_token'] == 'owner'
        assert not await router.state._backend.keys('model::*')
    finally:
        finish.set()
        await router.shutdown_task_queue()


@pytest.mark.asyncio
async def test_keyed_upload_reconnects_to_original_task(monkeypatch, tmp_path):
    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    checkpoint = create_checkpoint_manager('owner', client_type='twinkle').training_run_manager.get_base_dir() / 'saved'
    checkpoint.mkdir(parents=True)
    first_claim, allow_first_claim = asyncio.Event(), asyncio.Event()
    registered, continue_submission = asyncio.Event(), asyncio.Event()

    class PausedBackend(FakeBackend):
        async def set_nx(self, key, value, ttl=None):
            keyed_upload = key.startswith('future::req_upload_') and value['owner_token'] == 'owner'
            if keyed_upload and not first_claim.is_set():
                first_claim.set()
                await allow_first_claim.wait()
            created = await super().set_nx(key, value, ttl)
            if keyed_upload and created:
                registered.set()
                await continue_submission.wait()
            return created

    router = TrainingRouter.__new__(TrainingRouter)
    router.state = ServerState(backend=PausedBackend())
    router.replica_id = 'router'
    router._init_task_queue()
    finish = threading.Event()

    def transfer(*, report, **_):
        report(stage='uploading', uploaded_bytes=5, total_bytes=10)
        assert finish.wait(5)
        return 'commit-sha'

    upload = Mock(side_effect=transfer)
    monkeypatch.setattr(HubOperation, 'upload_checkpoint', upload)
    body = {'checkpoint_dir': str(checkpoint), 'hub_model_id': 'owner/saved', 'client_upload_id': 'operation-1'}

    async def submit(*, token='owner', **changes):
        return await router.native_request(http_request({**body, **changes}, token), 'upload_to_hub')

    try:
        raced = asyncio.create_task(submit())
        await asyncio.wait_for(first_claim.wait(), 3)
        first = asyncio.create_task(submit())
        await asyncio.wait_for(registered.wait(), 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        allow_first_claim.set()
        retry = await asyncio.wait_for(raced, 3)
        assert retry.status == 'pending'
        assert (await submit()).request_id == retry.request_id
        with pytest.raises(RequestRejectedError, match='different checkpoint or repository'):
            await submit(hub_model_id='owner/other')
        with pytest.raises(ResourceNotFoundError):
            await poll_future(router.state, retry.request_id, owner_token='another-tenant')
        other = await submit(token='another-tenant')
        assert other.request_id != retry.request_id
        assert (await settle(router.state, other.request_id))['status'] == 'failed'

        continue_submission.set()
        await asyncio.wait_for(poll_future(router.state, retry.request_id, after_progress=1), 3)
        assert (await submit()).request_id == retry.request_id
        finish.set()
        await settle(router.state, retry.request_id)
        completed = await submit()
        assert completed.request_id == retry.request_id
        assert completed.status == 'completed'
        assert completed.result == {'repo_id': 'owner/saved', 'revision': 'commit-sha'}
        upload.assert_called_once()
    finally:
        allow_first_claim.set()
        continue_submission.set()
        finish.set()
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
    router.config = TrainingConfig(models=[{'name': 'small', 'model_id': 'weights', 'backend': 'transformers', 'device_type': 'cuda', 'mode': 'lora', 'config': {'num_devices': 1, 'device_mesh': {'dp_size': 1}}}])
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


def test_router_schemas_match_model_runtime_routes():
    from fastapi import FastAPI
    from twinkle.server.model.tinker_handlers import _register_model_tinker_routes
    from twinkle.server.model.twinkle_handlers import _register_model_twinkle_routes
    from twinkle.server.model.pool.router import NATIVE_REQUESTS, TINKER_REQUESTS

    app = FastAPI()
    _register_model_tinker_routes(app, lambda: None)
    _register_model_twinkle_routes(app, lambda: None)
    for protocol, schemas in [('twinkle', NATIVE_REQUESTS), ('tinker', TINKER_REQUESTS)]:
        endpoints = {route.path.rsplit('/', 1)[-1] for route in app.routes
                     if route.path.startswith(f'/{protocol}/') and 'POST' in route.methods}
        assert endpoints == set(schemas)
