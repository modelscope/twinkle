# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from twinkle.dataset import DatasetMeta
from twinkle.protocol.serialize import deserialize_object, serialize_object
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientContext, ClientTransport


class _Response:

    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.url = 'http://server'
        self.text = ''

    def json(self):
        return self._payload


class _Session:

    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith('/create'):
            return _Response({'processor_id': 'pid:1'})
        return _Response({'result': 'ok'})

    def close(self):
        pass


def test_component_rpc_uses_transport_url_and_preserves_timeout_semantics() -> None:
    session = _Session()
    transport = ClientTransport(ClientContext(base_url='http://server', api_key='key'), session=session, timeout=90)

    assert create_remote_component('dataset', 'Dataset', transport=transport) == 'pid:1'
    assert call_remote_component('pid:1', 'check', transport=transport) == 'ok'
    assert call_remote_component('pid:1', 'check', None, transport=transport) == 'ok'

    assert session.calls[0][0] == 'http://server/api/v1/processor/twinkle/create'
    assert session.calls[1][0] == 'http://server/api/v1/processor/twinkle/call'
    assert session.calls[1][1]['timeout'] == 90
    assert session.calls[2][1]['timeout'] is None


def test_dataset_meta_data_slice_round_trips() -> None:
    for data_slice in (range(1, 9, 2), [1, 3, 5]):
        restored = deserialize_object(serialize_object(DatasetMeta(dataset_id='demo', data_slice=data_slice)))
        assert restored.dataset_id == 'demo'
        assert list(restored.data_slice) == list(data_slice)
