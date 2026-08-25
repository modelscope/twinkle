from twinkle.data_format import SamplingParams
from twinkle_client.sampler.vllm_sampler import vLLMSampler


def test_http_sampler_serializes_sampling_params_dataclass_once(monkeypatch):
    request = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'samples': []}

    def post(*, url, json_data):
        request['url'] = url
        request['body'] = json_data
        return Response()

    import twinkle_client.sampler.vllm_sampler as module
    monkeypatch.setattr(module, 'http_post', post)
    sampler = object.__new__(vLLMSampler)
    sampler.server_url = 'http://example/sampler/model/twinkle'
    sampler.sample([{'messages': []}], SamplingParams(max_tokens=4, num_samples=2))
    assert request['body']['sampling_params']['num_samples'] == 2
    assert 'num_samples' not in request['body']
