import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from tinker import types
from unittest.mock import AsyncMock, MagicMock, patch

from twinkle.server.model.tinker_handlers import _register_model_tinker_routes
from twinkle.server.deployment import twinkle_server_error_handler
from twinkle.server.exceptions import BatchSizeError, TwinkleServerError
from twinkle.server.task_queue.config import TaskQueueConfig
from twinkle.server.task_queue.mixin import TaskQueueMixin


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

    is_full_mode = False

    def __init__(self):
        self.model = MagicMock()
        self.state = MagicMock()
        self.state.get_model_metadata = AsyncMock(return_value={'base_model': 'test-model'})
        self.state.create_sampling_session = AsyncMock(return_value='session-123')

    async def _on_request_start(self, request):
        return 'token1'

    def get_adapter_name(self, adapter_name=None):
        return adapter_name

    def resolve_model_adapter_name(self, adapter_name):
        return adapter_name

    def assert_resource_exists(self, adapter_name):
        pass

    async def schedule_task(self, task, **kwargs):
        return await task()

    async def call_backend(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('backward', [False, True])
async def test_tinker_loss_metric_survives_sdk_reduction(backward):
    from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

    management = _SaveWeightsDummyManagement()
    management.data_world_size = 1
    management.set_resource_state = MagicMock()
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
@patch('twinkle.server.model.tinker_handlers.create_checkpoint_manager')
async def test_save_weights_for_sampler_path_mode_returns_path(mock_create_ckpt_mgr):
    """save_weights_for_sampler(name) mode: sampling_session_seq_id is None → returns path != None."""
    mock_ckpt_mgr = MagicMock()
    mock_ckpt_mgr.get_ckpt_name.return_value = 'step-1'
    mock_ckpt_mgr.get_save_dir.return_value = '/tmp/save_dir'
    mock_ckpt_mgr.save.return_value = 'twinkle://model1/sampler_weights/20260101_000000'
    mock_create_ckpt_mgr.return_value = mock_ckpt_mgr

    management = _SaveWeightsDummyManagement()
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)

    body = types.SaveWeightsForSamplerRequest(
        model_id='model1',
        path='step-1',
        sampling_session_seq_id=None,  # path mode
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/save_weights_for_sampler')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert response.path == 'twinkle://model1/sampler_weights/20260101_000000'
    assert response.sampling_session_id == 'session-123'


@pytest.mark.asyncio
@patch('twinkle.server.model.tinker_handlers.create_checkpoint_manager')
async def test_save_weights_for_sampler_session_mode_returns_none_path(mock_create_ckpt_mgr):
    """save_weights_and_get_sampling_client() mode: sampling_session_seq_id is set → returns path == None."""
    mock_ckpt_mgr = MagicMock()
    mock_ckpt_mgr.get_ckpt_name.return_value = 'step-1'
    mock_ckpt_mgr.get_save_dir.return_value = '/tmp/save_dir'
    mock_ckpt_mgr.save.return_value = 'twinkle://model1/sampler_weights/20260101_000000'
    mock_create_ckpt_mgr.return_value = mock_ckpt_mgr

    management = _SaveWeightsDummyManagement()
    app = FastAPI()
    _register_model_tinker_routes(app, lambda: management)

    body = types.SaveWeightsForSamplerRequest(
        model_id='model1',
        sampling_session_seq_id=0,  # session mode
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/save_weights_for_sampler')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert response.path is None
    assert response.sampling_session_id == 'session-123'
