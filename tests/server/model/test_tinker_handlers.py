import pytest
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
