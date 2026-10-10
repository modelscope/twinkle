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
async def test_sampler_save_retains_names_and_forwards_backend_kwargs(monkeypatch, tmp_path):
    from pathlib import Path
    from twinkle.server.checkpoint.twinkle import TwinkleCheckpointManager, TwinkleTrainingRunManager
    import twinkle.server.model.twinkle_handlers as handlers

    class Management(_SchedulingManagement):
        def save(self, name, output_dir, **kwargs):
            path = Path(output_dir) / name
            path.mkdir()
            self.model_calls.append((name, kwargs))
            (path / 'weights').write_text(str(len(self.model_calls)))
            return str(path)

    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    runs = TwinkleTrainingRunManager('token')
    runs._write_info('session-adapter', {'base_model': 'test-model', 'is_lora': True})
    manager = TwinkleCheckpointManager('token', runs)
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    service = Management()
    app = FastAPI()
    _register_model_twinkle_routes(app, lambda: service)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/twinkle/save')
    request = Request({'type': 'http', 'headers': []})
    request.state.session_id = 'session'

    async def save(name):
        body = types.SaveRequest(adapter_name='adapter', name=name, is_sampler=True,
                                 save_optimizer=True, consumed_train_samples=7)
        return await endpoint(request, body, service)

    x, y, live = await save('x'), await save('y'), await save(None)
    x_dir = Path(x.result['checkpoint_dir'])
    assert x.result['twinkle_path'].endswith('/sampler_weights/x')
    assert (x_dir / 'weights').read_text() == '1'
    assert Path(y.result['checkpoint_dir'], 'weights').read_text() == '2'
    assert Path(live.result['checkpoint_dir']).name == 'latest'
    overwritten = await save('x')
    assert overwritten.result['twinkle_path'] == x.result['twinkle_path']
    assert (x_dir / 'weights').read_text() == '4'
    assert service.model_calls[-1][1] == {'adapter_name': 'session-adapter',
                                        'save_optimizer': True, 'consumed_train_samples': 7}
