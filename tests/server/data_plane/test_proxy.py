from __future__ import annotations

import pytest

from twinkle.server.data_plane.proxy import DataPlaneProxy
from twinkle_client.http.headers import H_AUTH, H_AUTH_TWINKLE, H_REQUEST_ID
from twinkle_client.types import DataRef


class _Response:

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _Client:

    def __init__(self):
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith('/get'):
            return _Response({'rows': [{'value': 1}]})
        return _Response({
            'ref_id': 'output',
            'size': 1,
            'fields': ['value'],
            'kind': 'model-output',
        })


@pytest.mark.asyncio
async def test_proxy_routes_by_data_ref_without_tenant_identity() -> None:
    proxy = DataPlaneProxy.__new__(DataPlaneProxy)
    proxy.base_url = 'http://data-plane'
    proxy.client = _Client()
    ref = DataRef(ref_id='input', size=1, fields=['value'])

    assert await proxy.get(ref, fields=['value']) == [{'value': 1}]
    output = await proxy.put([{'value': 2}], kind='model-output')
    appended = await proxy.append(ref, [{'value': 3}])

    assert output.ref_id == 'output'
    assert appended.ref_id == 'output'
    get_headers = proxy.client.calls[0][1]['headers']
    put_headers = proxy.client.calls[1][1]['headers']
    append_headers = proxy.client.calls[2][1]['headers']
    assert proxy.client.calls[0][1]['json']['fields'] == ['value']
    assert get_headers[H_REQUEST_ID] == 'data-ref-input'
    assert put_headers[H_REQUEST_ID] == 'data-put-model-output'
    assert append_headers[H_REQUEST_ID] == 'data-append-input'
    assert get_headers[H_AUTH] == get_headers[H_AUTH_TWINKLE] == ''
    assert put_headers[H_AUTH] == put_headers[H_AUTH_TWINKLE] == ''
    assert append_headers[H_AUTH] == append_headers[H_AUTH_TWINKLE] == ''
