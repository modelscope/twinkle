import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from tinker import types
from unittest.mock import ANY, AsyncMock, MagicMock

from twinkle.server.model.app import ModelManagement
from twinkle.server.model.tinker_handlers import _register_model_tinker_routes
from twinkle.server.deployment import twinkle_server_error_handler
from twinkle.server.exceptions import BatchSizeError, TwinkleServerError
from twinkle.server.config.task_queue import TaskQueueConfig
from twinkle.server.task_queue.mixin import TaskQueueMixin
from twinkle.server.model.runtime import TrainingExecutionState


class _DummyManagement:

    def __init__(self, data_world_size=2):
        self.scheduled = []
        self.data_world_size = data_world_size
        self._task_queue_config = TaskQueueConfig(enabled=True)
        self._rate_limiter = AsyncMock()
        self._rate_limiter.check_and_record.return_value = (True, '')

    async def _on_request_start(self, request):
        return 'token1'

    async def schedule_task(self, task, **kwargs):
        await TaskQueueMixin._perform_preflight_checks(
            self,
            model_id=kwargs.get('model_id'),
            token=kwargs.get('token'),
            input_tokens=kwargs.get('input_tokens', 0),
            batch_size=kwargs.get('batch_size'),
            data_world_size=kwargs.get('data_world_size'),
            batch_size_multiple=kwargs.get('batch_size_multiple'),
        )
        self.scheduled.append(kwargs)
        return {'request_id': 'req1', 'model_id': kwargs.get('model_id')}


def _datum(*, dpo=False):
    loss_fn_inputs = {}
    if dpo:
        loss_fn_inputs['ref_logps'] = types.TensorData(data=[-0.1, -0.2], dtype='float32', shape=[2])
    return types.Datum(model_input=types.ModelInput.from_ints([1, 2]), loss_fn_inputs=loss_fn_inputs)


@pytest.mark.asyncio
@pytest.mark.parametrize('data_world_size,batch_size', [(1, 2), (2, 4)])
async def test_tinker_dpo_forward_backward_requires_per_dp_pairs(data_world_size, batch_size):
    management = _DummyManagement(data_world_size=data_world_size)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)

    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum(dpo=True) for _ in range(batch_size)],
            loss_fn='importance_sampling',
        ),
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert response == {'request_id': 'req1', 'model_id': 'model1'}
    assert management.scheduled[-1]['batch_size'] == batch_size
    assert management.scheduled[-1]['data_world_size'] == data_world_size
    assert management.scheduled[-1]['batch_size_multiple'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('batch_size', [1, 3])
async def test_tinker_rl_accepts_odd_batches_on_one_data_rank(batch_size):
    management = _DummyManagement(data_world_size=1)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum() for _ in range(batch_size)], loss_fn='importance_sampling'),
    )
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')

    await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)

    assert management.scheduled[-1]['batch_size'] == batch_size
    assert management.scheduled[-1]['batch_size_multiple'] is None


@pytest.mark.asyncio
async def test_tinker_rl_does_not_require_pairs_on_multiple_data_ranks():
    management = _DummyManagement(data_world_size=2)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum(), _datum()], loss_fn='importance_sampling'),
    )
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')

    await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)

    assert management.scheduled[-1]['batch_size_multiple'] is None


@pytest.mark.asyncio
async def test_tinker_rl_still_rejects_batches_smaller_than_data_world_size():
    management = _DummyManagement(data_world_size=2)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(data=[_datum()], loss_fn='importance_sampling'),
    )
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')

    with pytest.raises(BatchSizeError, match='must be >= data world size'):
        await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)

    assert management.scheduled == []


@pytest.mark.parametrize('loss_fn', ['ppo', 'cispo', 'dro'])
def test_tinker_unsupported_loss_is_http_400_before_enqueue(loss_fn):
    management = _DummyManagement(data_world_size=1)
    app = FastAPI()
    app.add_exception_handler(TwinkleServerError, twinkle_server_error_handler)
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(data=[_datum()], loss_fn=loss_fn),
    )

    response = TestClient(app).post('/tinker/forward_backward', json=body.model_dump(mode='json'))

    assert response.status_code == 400
    assert response.json()['category'] == 'user'
    assert f"Unsupported Tinker loss_fn '{loss_fn}'" in response.json()['error']
    assert management.scheduled == []
    management._rate_limiter.check_and_record.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('data_world_size,batch_size', [(1, 1), (1, 3), (2, 2), (2, 6)])
async def test_tinker_dpo_rejects_incomplete_per_rank_pairs_before_enqueue(data_world_size, batch_size):
    management = _DummyManagement(data_world_size=data_world_size)
    # Pair validation must also hold when rate limiting / task queue checks are disabled.
    management._task_queue_config = TaskQueueConfig(enabled=False)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum(dpo=True) for _ in range(batch_size)], loss_fn='importance_sampling'),
    )
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')

    with pytest.raises(BatchSizeError, match='complete chosen/rejected pairs'):
        await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)

    assert management.scheduled == []


def test_tinker_rejects_mixed_dpo_and_rl_before_enqueue():
    management = _DummyManagement(data_world_size=1)
    app = FastAPI()
    app.add_exception_handler(TwinkleServerError, twinkle_server_error_handler)
    _register_model_tinker_routes(app, lambda: management)
    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum(dpo=True), _datum()], loss_fn='importance_sampling'),
    )

    response = TestClient(app).post('/tinker/forward_backward', json=body.model_dump(mode='json'))

    assert response.status_code == 400
    assert 'cannot mix DPO and RL' in response.json()['error']
    assert management.scheduled == []


class _SaveWeightsDummyManagement:
    """Dummy management that actually executes the task to test save_weights_for_sampler logic."""

    train_mode = 'lora'
    is_full_mode = ModelManagement.is_full_mode

    def __init__(self):
        self.training_state = TrainingExecutionState(grad_ready=True)
        self.model = MagicMock()
        self.state = MagicMock()
        self.state.get_model_metadata = AsyncMock(return_value={'base_model': 'test-model'})
        self.state.create_sampling_session = AsyncMock(return_value='session-123')

    async def _on_request_start(self, request):
        return 'token1'

    def get_training_state(self, training_id):
        return self.training_state

    resolve_model_adapter_name = ModelManagement.resolve_model_adapter_name

    def assert_resource_exists(self, adapter_name):
        pass

    async def schedule_task(self, task, **kwargs):
        return await task()

    async def call_backend(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('full_mode,fail_backend', [(False, False), (True, False), (False, True)],
                         ids=['lora', 'full-model', 'backend-failure'])
async def test_training_save_records_metadata_only_after_backend_success(monkeypatch, tmp_path, fail_backend, full_mode):
    from pathlib import Path
    import twinkle.server.model.tinker_handlers as handlers
    from twinkle.server.checkpoint.tinker import TinkerCheckpointManager, TinkerTrainingRunManager

    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    runs = TinkerTrainingRunManager('token1')
    runs._write_info('model1', {'base_model': 'test-model'})
    manager = TinkerCheckpointManager('token1', runs)
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    management = _SaveWeightsDummyManagement()
    management.train_mode = 'full' if full_mode else 'lora'

    def save(name, output_dir, **kwargs):
        assert not manager.list_checkpoints('model1').checkpoints
        assert kwargs == {'adapter_name': '' if full_mode else 'model1', 'save_optimizer': True}
        if fail_backend:
            raise RuntimeError('backend save failed')
        directory = Path(output_dir) / name
        directory.mkdir(parents=True)
        (directory / 'weights').write_text('training')
        return str(directory)

    management.model.save.side_effect = save
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/tinker/save_weights')
    body = types.SaveWeightsRequest(model_id='model1', path='step/one')
    request = Request({'type': 'http', 'headers': []})
    if fail_backend:
        with pytest.raises(RuntimeError, match='backend save failed'):
            await endpoint(request, body, management)
        assert not manager.list_checkpoints('model1').checkpoints
        assert 'last_checkpoint' not in runs._read_info('model1')
    else:
        response = await endpoint(request, body, management)
        assert response.path == 'twinkle://model1/weights/step_one'
        assert response.type == 'save_weights'
        assert manager.get('model1', 'weights/step_one').size_bytes > 0


@pytest.mark.asyncio
@pytest.mark.parametrize('path,output_dir', [('org/model', None), ('twinkle://model1/weights/step', 'local')])
async def test_load_weights_resolves_location_and_preserves_optimizer_flag(monkeypatch, tmp_path, path, output_dir):
    import twinkle.server.model.tinker_handlers as handlers
    from twinkle.server.checkpoint.tinker import TinkerCheckpointManager, TinkerTrainingRunManager
    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    manager = TinkerCheckpointManager('token1', TinkerTrainingRunManager('token1'))
    if output_dir:
        manager.save('model1', 'step')
        output_dir = manager.get_save_dir('model1')
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    management = _SaveWeightsDummyManagement()
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/tinker/load_weights')
    response = await endpoint(Request({'type': 'http', 'headers': []}),
                              types.LoadWeightsRequest(model_id='model1', path=path, optimizer=True), management)
    management.model.tinker_load.assert_called_once_with(
        checkpoint_name='step' if output_dir else path, output_dir=output_dir,
        load_optimizer=True, adapter_name='model1')
    assert management.training_state.grad_ready is False
    assert response.path == path and response.type == 'load_weights'


@pytest.mark.asyncio
@pytest.mark.parametrize('backward', [False, True])
async def test_tinker_loss_metric_survives_sdk_reduction(backward):
    from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

    management = _SaveWeightsDummyManagement()
    management.data_world_size = 1
    outputs = [{'logprobs': types.TensorData(data=[-0.25, -0.5], dtype='float32', shape=[2])}]
    management.model.tinker_forward_only.return_value = (outputs, 0.375)
    management.model.tinker_forward_backward.return_value = (outputs, 0.375)
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    if backward:
        body = types.ForwardBackwardRequest(
            model_id='model1',
            forward_backward_input=types.ForwardBackwardInput(data=[_datum()], loss_fn='cross_entropy'),
        )
        path = '/tinker/forward_backward'
    else:
        body = types.ForwardRequest(
            model_id='model1',
            forward_input=types.ForwardBackwardInput(data=[_datum()], loss_fn='cross_entropy'),
        )
        path = '/tinker/forward'
    route = next(route for route in app.routes if getattr(route, 'path', None) == path)
    response = await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)

    # Use the pinned SDK's real combiner: unsupported reduction names are silently dropped.
    combined = combine_fwd_bwd_output_results([response])
    assert combined.metrics['loss:mean'] == pytest.approx(0.375)


@pytest.mark.asyncio
@pytest.mark.parametrize('name,sequence_id', [('step-1', None), (None, 0)], ids=['path', 'session'])
@pytest.mark.parametrize('full_mode', [False, True], ids=['lora', 'full-model'])
async def test_sampler_save_materializes_weights_and_returns_sdk_mode(monkeypatch, tmp_path, name, sequence_id, full_mode):
    from pathlib import Path
    from twinkle.server.checkpoint import create_checkpoint_manager
    import twinkle.server.model.tinker_handlers as handlers
    monkeypatch.setattr('twinkle.server.checkpoint.training_run_manager.TWINKLE_DEFAULT_SAVE_DIR', str(tmp_path))
    manager = create_checkpoint_manager('token1', client_type='tinker')
    manager.training_run_manager._write_info('model1', {'base_model': 'test-model'})
    monkeypatch.setattr(handlers, 'create_checkpoint_manager', lambda *args, **kwargs: manager)
    management = _SaveWeightsDummyManagement()
    management.train_mode = 'full' if full_mode else 'lora'

    def save(name, output_dir, **kwargs):
        directory = Path(output_dir) / name
        directory.mkdir(parents=True)
        (directory / 'weights').write_text('sampler')

    management.model.save.side_effect = save
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)
    body = types.SaveWeightsForSamplerRequest(model_id='model1', path=name, sampling_session_seq_id=sequence_id)
    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/save_weights_for_sampler')
    response = await route.endpoint(Request({'type': 'http', 'headers': []}), body, management)
    management.model.save.assert_called_once_with(
        name=name or 'latest', output_dir=ANY, adapter_name='' if full_mode else 'model1', save_optimizer=False)
    management.state.create_sampling_session.assert_awaited_once()
    payload = management.state.create_sampling_session.call_args.args[0]
    uri = payload['model_path']
    assert payload == {**body.model_dump(), 'model_path': uri, 'base_model': 'test-model'}
    checkpoint_id = manager.parse_path(uri).checkpoint_id
    assert Path(manager.get_ckpt_dir('model1', checkpoint_id), 'weights').read_text() == 'sampler'
    assert response.path == (uri if sequence_id is None else None)
    if name:
        assert uri == 'twinkle://model1/sampler_weights/step-1'
    assert response.sampling_session_id == 'session-123'
