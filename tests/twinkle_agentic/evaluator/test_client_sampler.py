from twinkle.data_format import SamplingParams
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.sampler.vllm_sampler import vLLMSampler


class _Response:

    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200

    def json(self):
        return self._payload


class _Session:

    def __init__(self, request):
        self.request = request

    def post(self, url, *, data=None, json=None, **_kwargs):
        self.request['url'] = url
        if data is not None:
            import json as json_module
            self.request['body'] = json_module.loads(data)
        else:
            self.request['body'] = json
        if url.endswith('/create'):
            return _Response({})
        return _Response({'request_id': 'test', 'status': 'completed', 'result': {'samples': []}})

    def close(self):
        pass


def test_http_sampler_serializes_sampling_params_dataclass_once():
    request = {}
    transport = ClientTransport(
        ClientContext(base_url='http://example', api_key='test'),
        session=_Session(request),
    )
    sampler = vLLMSampler('model', transport=transport)
    sampler.sample([{'messages': []}], SamplingParams(max_tokens=4, num_samples=2))
    assert request['body']['sampling_params']['num_samples'] == 2
    assert 'num_samples' not in request['body']
