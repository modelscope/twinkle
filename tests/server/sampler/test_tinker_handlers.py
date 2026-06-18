import pytest
from fastapi import FastAPI
from starlette.requests import Request
from tinker import types

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.server.sampler.tinker_handlers import _register_tinker_sampler_routes


class _DummyState:

    async def get_sampling_session(self, sampling_session_id: str):
        return {
            'base_model': 'mock-model',
            'model_path': None,
        }


class _DummySampler:

    def __init__(self):
        self.adapter_paths = []

    def set_template(self, *args, **kwargs):
        return None

    def reset_prefix_cache(self):
        return None

    def sample(self, inputs, sampling_params=None, adapter_name='', *, adapter_path=None, **kwargs):
        self.adapter_paths.append(adapter_path)
        return [
            SampleResponse(
                sequences=[SampledSequence(
                    stop_reason='length',
                    tokens=[1, 2],
                    logprobs=[[(1, -0.1)], [(2, -0.2)]],
                )])
        ]


class _DummyManagement:

    def __init__(self):
        self.model_id = 'mock-model'
        self.state = _DummyState()
        self.sampler = _DummySampler()

    async def _on_request_start(self, request):
        return 'EMPTY_TOKEN'

    async def schedule_task(self, task, **kwargs):
        return await task()


@pytest.mark.asyncio
async def test_tinker_asample_allows_base_model_session_without_model_path():
    management = _DummyManagement()
    app = FastAPI()
    _register_tinker_sampler_routes(app, lambda: management)

    body = types.SampleRequest(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        sampling_params=types.SamplingParams(max_tokens=2),
        sampling_session_id='base-session',
        base_model='mock-model',
    )

    route = next(route for route in app.routes if getattr(route, 'path', None) == '/tinker/asample')
    request = Request({'type': 'http', 'headers': []})
    response = await route.endpoint(request, body, management)

    assert isinstance(response, types.SampleResponse)
    assert management.sampler.adapter_paths == [None]
