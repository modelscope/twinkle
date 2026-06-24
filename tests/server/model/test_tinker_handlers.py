import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from starlette.requests import Request
from tinker import types

from twinkle.server.model.tinker_handlers import _register_tinker_routes


class _DummyManagement:

    def __init__(self):
        self.scheduled = []
        self.data_world_size = 2

    async def _on_request_start(self, request):
        return 'token1'

    async def schedule_task(self, task, **kwargs):
        self.scheduled.append(kwargs)
        return {'request_id': 'req1', 'model_id': kwargs.get('model_id')}


def _datum():
    return types.Datum(model_input=types.ModelInput.from_ints([1, 2]), loss_fn_inputs={})


@pytest.mark.asyncio
async def test_tinker_dpo_forward_backward_requires_per_dp_pairs():
    management = _DummyManagement()
    app = FastAPI()
    _register_tinker_routes(app, lambda: management)

    body = types.ForwardBackwardRequest(
        model_id='model1',
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum(), _datum()],
            loss_fn='importance_sampling',
        ),
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/forward_backward')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert response == {'request_id': 'req1', 'model_id': 'model1'}
    assert management.scheduled[-1]['batch_size'] == 2
    assert management.scheduled[-1]['data_world_size'] == 2
    assert management.scheduled[-1]['batch_size_multiple'] == 2


class _SaveWeightsDummyManagement:
    """Dummy management that actually executes the task to test save_weights_for_sampler logic."""

    def __init__(self):
        self.model = MagicMock()
        self.state = MagicMock()
        self.state.get_model_metadata = AsyncMock(return_value={'base_model': 'test-model'})
        self.state.create_sampling_session = AsyncMock(return_value='session-123')

    async def _on_request_start(self, request):
        return 'token1'

    def get_adapter_name(self, adapter_name=None):
        return adapter_name

    def assert_resource_exists(self, adapter_name):
        pass

    async def schedule_task(self, task, **kwargs):
        # Actually execute the task to test response logic
        return await task()


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
    _register_tinker_routes(app, lambda: management)

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
    _register_tinker_routes(app, lambda: management)

    body = types.SaveWeightsForSamplerRequest(
        model_id='model1',
        sampling_session_seq_id=0,  # session mode
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/save_weights_for_sampler')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert response.path is None
    assert response.sampling_session_id == 'session-123'
