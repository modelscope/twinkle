"""A rejected or timed-out request must not disable unrelated sampling."""
import json

import pytest
from ray.exceptions import GetTimeoutError
from starlette.requests import Request

from twinkle.protocol.types import SampleRequest
from .sampling_test_helpers import _tokens, make_service


@pytest.mark.asyncio
@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
async def test_invalid_checkpoint_rejects_request_without_disabling_sampling(make_service, tmp_path, dialect):
    service = make_service()
    ref = await service.submit(dialect, path=str(tmp_path / 'missing'))
    failed = await service.result(ref)
    assert failed['failure'].reason_code == 'request_rejected'
    assert failed['failure'].attribution == 'user'
    assert _tokens(await service.result(await service.submit(dialect)), dialect) == [7]


@pytest.mark.asyncio
async def test_ordinary_backend_timeout_does_not_disable_replica(make_service):
    service = make_service()
    sample = service.sampler.sample
    service.sampler.sample = lambda *a, **k: (_ for _ in ()).throw(GetTimeoutError('one request timed out'))
    with pytest.raises(GetTimeoutError):
        await service.call_backend(service.sampler.sample)
    service.sampler.sample = sample
    assert _tokens(await service.result(await service.submit('twinkle')), 'twinkle') == [7]


@pytest.mark.asyncio
async def test_stream_route_resolves_adapter_uri(make_service, tmp_path):
    service = make_service()
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'r': 1}))
    service.sampler.reset_prefix_cache = lambda: None
    service.sampler._actors = []
    request = Request({'type': 'http', 'headers': []})
    response = await service.endpoints['/twinkle/sample_stream'](
        request, SampleRequest(inputs=[{'input_ids': [7]}], adapter_uri=str(tmp_path)), service)
    rows = [json.loads(row) async for row in response.body_iterator]
    assert rows[0]['error_code'] == 503


@pytest.mark.asyncio
async def test_sampling_timeout_keeps_capacity_without_blocking_other_readers(make_service):
    import asyncio
    from .sampling_test_helpers import _until
    service = make_service(timeout=0.05, limit=2)
    service.sampler.release.clear()
    timed_out = await service.submit('tinker')
    await _until(service.sampler.entered.is_set)
    assert (await service.result(timed_out))['failure'].reason_code == 'execution_timeout'
    submitting = asyncio.create_task(service.submit('twinkle'))
    await _until(lambda: service.sampler.peak == 2)
    service.sampler.release.set()
    assert _tokens(await service.result(await submitting), 'twinkle') == [7]
    await _until(lambda: service._compute_worker._detached_reserved == 0)
    await service.check_health()
