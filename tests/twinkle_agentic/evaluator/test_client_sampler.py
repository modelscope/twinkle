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
        import json as json_module
        self.request['url'] = url
        body = json_module.loads(data) if data is not None else json
        self.request['body'] = body
        if url.endswith('/create'):
            return _Response({})
        # 012aff34 complete-Trajectory contract: a normal (non token_only) response must return
        # one sample per input, each sequence carrying a complete new_input_feature.
        samples = [{'sequences': [{'stop_reason': 'stop', 'tokens': [],
                                   'new_input_feature': dict(item)}]} for item in body['inputs']]
        return _Response({'request_id': 'test', 'status': 'completed', 'result': {'samples': samples}})

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
