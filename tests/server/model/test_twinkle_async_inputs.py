from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.requests import Request

import twinkle.protocol.types as types
from twinkle.server.model.data_plane_inputs import model_result_rows
from twinkle.server.model.twinkle_handlers import _register_model_twinkle_routes


def test_model_result_rows_keeps_one_output_row_per_sample() -> None:
    assert model_result_rows(
        {
            'logps': [[-1.0], [-2.0]],
            'loss': 0.25
        },
        batch_size=2,
    ) == [
        {
            'logps': [-1.0],
            'loss': 0.25
        },
        {
            'logps': [-2.0],
            'loss': 0.25
        },
    ]


class _SchedulingManagement:

    def __init__(self):
        self.data_world_size = 2
        self.scheduled = []
        self.model_calls = []
        self.model = self
        self.data_plane = self
        self.rows = {
            'data-a': [{
                'train_input': {
                    'input_ids': [index]
                },
                'sampled_logprobs': [-0.1],
                'advantage': 1.0,
            } for index in range(4)],
            'data-b': [{
                'train_input': {
                    'input_ids': [index]
                },
                'sampled_logprobs': [-0.2],
                'advantage': -1.0,
            } for index in range(4, 8)],
        }

    async def _on_request_start(self, _request):
        return 'token'

    def assert_resource_exists(self, _adapter_name):
        return None

    def resolve_model_adapter_name(self, adapter_name):
        return adapter_name

    def forward_backward(self, *, inputs, adapter_name, **kwargs):
        self.model_calls.append((inputs, adapter_name, kwargs))
        return {'loss': 1.0}

    async def get(self, ref, *, fields=None):
        rows = self.rows[ref.ref_id]
        if fields is None:
            return rows
        return [{field: row[field] for field in fields} for row in rows]

    async def submit_and_peek(self, coro_factory, *, model_id=None, token=None, task_type=None, **schedule_kwargs):
        self.scheduled.append(schedule_kwargs)
        result = await coro_factory()
        from twinkle.protocol.types.lifecycle import TaskEnvelope
        return TaskEnvelope(request_id='req-test', status='completed', result=result)

    async def call_backend(self, fn, /, *args, admit=True, **kwargs):
        return fn(*args, **kwargs)


@pytest.mark.asyncio
async def test_forward_backward_resolves_multiple_data_refs_and_field_kwargs() -> None:
    management = _SchedulingManagement()
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: management)
    route = next(
        route for route in app.routes if getattr(route, 'path', None) == '/twinkle/forward_backward_from_data_plane')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.DataPlaneForwardRequest(
        adapter_name='adapter',
        input_refs=[
            types.DataRef(ref_id='data-a', size=4, num_tokens=4),
            types.DataRef(ref_id='data-b', size=4, num_tokens=4),
        ],
        input_field='train_input',
        kwarg_fields={
            'old_logps': 'sampled_logprobs',
            'advantages': 'advantage',
        },
    )

    await route.endpoint(request, body, management)

    assert management.scheduled[-1]['batch_size'] == 8
    assert management.scheduled[-1]['data_world_size'] == 2
    inputs, adapter_name, forwarded_kwargs = management.model_calls[-1]
    assert adapter_name == 'session-adapter'
    assert [row['input_ids'] for row in inputs] == [[index] for index in range(8)]
    assert forwarded_kwargs['old_logps'] == [[-0.1]] * 4 + [[-0.2]] * 4
    assert forwarded_kwargs['advantages'] == [1.0] * 4 + [-1.0] * 4


@pytest.mark.asyncio
async def test_forward_backward_binds_nested_dpo_ref_logps_without_coercion() -> None:
    management = _SchedulingManagement()
    management.rows['dpo'] = [
        {
            'input_ids': [1, 2, 3],
            'labels': [-100, 2, 3],
            'ref_logps': [-0.1, -0.2, -0.3],
        },
        {
            'input_ids': [1, 4, 5],
            'labels': [-100, 4, 5],
            'ref_logps': [-0.4, -0.5, -0.6],
        },
    ]
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: management)
    route = next(
        route for route in app.routes if getattr(route, 'path', None) == '/twinkle/forward_backward_from_data_plane')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.DataPlaneForwardRequest(
        adapter_name='adapter',
        input_refs=[types.DataRef(ref_id='dpo', size=2, num_tokens=6)],
        kwarg_fields={'ref_outputs.logps': 'ref_logps'},
    )

    await route.endpoint(request, body, management)

    inputs, adapter_name, forwarded_kwargs = management.model_calls[-1]
    assert adapter_name == 'session-adapter'
    assert [row['input_ids'] for row in inputs] == [[1, 2, 3], [1, 4, 5]]
    assert forwarded_kwargs['ref_outputs']['logps'] == [
        [-0.1, -0.2, -0.3],
        [-0.4, -0.5, -0.6],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('is_sampler,save_optimizer', [(False, False), (True, True)],
                         ids=['training-no-optimizer', 'sampler-with-optimizer'])
@pytest.mark.parametrize('full_mode', [False, True], ids=['lora', 'full-model'])
async def test_save_forwards_backend_options_and_returns_public_path(
        monkeypatch, tmp_path, is_sampler, save_optimizer, full_mode):
    from pathlib import Path
    from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager
    from twinkle.server.model.app import ModelManagement
    import twinkle.server.model.twinkle_handlers as handlers

    class Management(_SchedulingManagement):
        is_full_mode = ModelManagement.is_full_mode
        resolve_model_adapter_name = ModelManagement.resolve_model_adapter_name

        def save(self, name, output_dir, **kwargs):
            path = Path(output_dir) / name
            path.mkdir(parents=True, exist_ok=True)
            self.model_calls.append((name, kwargs))
            (path / 'weights').write_text(str(len(self.model_calls)))
            return str(path)

    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    runs = TwinkleTrainingRunManager('token')
    runs._write_info('session-adapter', {'base_model': 'test-model', 'is_lora': True})
    manager = TwinkleCheckpointManager('token', runs)
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    service = Management()
    service.train_mode = 'full' if full_mode else 'lora'
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: service)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/twinkle/save')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'

    body = types.SaveRequest(adapter_name='adapter', name='step/one', is_sampler=is_sampler,
                             save_optimizer=save_optimizer, consumed_train_samples=7)
    response = await endpoint(request, body, service)
    weights_type = 'sampler_weights' if is_sampler else 'weights'
    directory = manager.get_ckpt_dir('session-adapter', f'{weights_type}/step_one')
    assert response.result == {'twinkle_path': f'twinkle://session-adapter/{weights_type}/step_one',
                               'checkpoint_dir': str(directory)}
    assert (directory / 'weights').read_text() == '1'
    assert service.model_calls == [('step_one', {'adapter_name': '' if full_mode else 'session-adapter',
                                                'save_optimizer': save_optimizer, 'consumed_train_samples': 7})]


@pytest.mark.asyncio
async def test_training_save_failure_does_not_publish_checkpoint(monkeypatch, tmp_path):
    from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager
    import twinkle.server.model.twinkle_handlers as handlers

    class Management(_SchedulingManagement):
        def save(self, **kwargs):
            raise RuntimeError('backend save failed')

    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    runs = TwinkleTrainingRunManager('token')
    runs._write_info('session-adapter', {'base_model': 'test-model'})
    manager = TwinkleCheckpointManager('token', runs)
    before = runs._read_info('session-adapter')
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    service = Management()
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: service)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/twinkle/save')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'
    body = types.SaveRequest(adapter_name='adapter', name='failed')

    with pytest.raises(RuntimeError, match='backend save failed'):
        await endpoint(request, body, service)
    assert manager.list_checkpoints('session-adapter').checkpoints == []
    assert runs._read_info('session-adapter') == before


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['load', 'resume_from_checkpoint', 'upload_to_hub'])
@pytest.mark.parametrize('local', [False, True], ids=['external', 'checkpoint-uri'])
async def test_checkpoint_consumers_keep_backend_path_and_options(monkeypatch, tmp_path, operation, local):
    from unittest.mock import MagicMock
    import twinkle.server.model.twinkle_handlers as handlers
    from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager

    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    manager = TwinkleCheckpointManager('token', TwinkleTrainingRunManager('token'))
    path = manager.save('run', 'step') if local else 'org/model'
    directory = str(manager.get_ckpt_dir('run', 'weights/step')) if local else path
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    service = _SchedulingManagement()
    service.backend = 'transformers'
    service.model = MagicMock()
    service.submit_background_and_peek = service.submit_and_peek
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: service)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == f'/twinkle/{operation}')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'

    if operation == 'load':
        body = types.LoadRequest(adapter_name='adapter', name=path, load_optimizer=True, strict=False)
    elif operation == 'resume_from_checkpoint':
        body = types.ResumeFromCheckpointRequest(adapter_name='adapter', name=path, resume_only_model=True)
        service.model.resume_from_checkpoint.return_value = {'step': 7}
    else:
        body = types.UploadToHubRequest(checkpoint_dir=path, hub_model_id='org/output', hub_token='hub-token')
    response = await endpoint(request, body, service)

    if operation == 'load':
        service.model.load.assert_called_once_with(
            name='step' if local else path, output_dir=manager.get_save_dir('run') if local else None,
            load_optimizer=True, adapter_name='session-adapter', token='token', strict=False)
    elif operation == 'resume_from_checkpoint':
        service.model.resume_from_checkpoint.assert_called_once_with(
            directory, resume_only_model=True, adapter_name='session-adapter')
        assert response.result == {'result': {'step': 7}}
    else:
        service.model.upload_to_hub.assert_called_once_with(
            checkpoint_dir=directory, hub_model_id='org/output', hub_token='hub-token', async_upload=False)
