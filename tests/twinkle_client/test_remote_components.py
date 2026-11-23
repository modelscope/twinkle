# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import subprocess
import sys
from types import SimpleNamespace

import json
import pytest
from torch.utils.data import IterableDataset as TorchIterableDataset

from twinkle.dataset import DatasetMeta
from twinkle.protocol.serialize import deserialize_object, serialize_object
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset, IterableDataset, IterablePackingDataset, LazyDataset, PackingDataset
from twinkle_client.http import ClientContext, ClientTransport
from twinkle_client.http.context import clear_default_transport, set_default_transport
from twinkle_client.manager import TwinkleClient


class _Session:

    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        body = json.loads(kwargs['data'])
        self.calls.append((url, body, kwargs['timeout']))
        result = {'processor_id': f"pid:{body['class_type']}"} if url.endswith('/create') else {'result': 'ok'}
        return SimpleNamespace(ok=True, status_code=200, url=url, text='', json=lambda: result)

    def close(self) -> None:
        pass


def test_remote_components_preserve_urls_wire_arguments_and_timeouts() -> None:
    session = _Session()
    transport = ClientTransport(ClientContext(base_url='http://server', api_key='key'), session=session, timeout=90)

    dataset = Dataset(transport=transport)
    loader = DataLoader(dataset)
    assert dataset.processor_id == 'pid:Dataset'
    assert dataset.check(flag=True) == 'ok'
    assert dataset.encode(timeout=None) == 'ok'
    assert loader.get_state() == 'ok'

    prefix = 'http://server/api/v1/processor/twinkle'
    assert session.calls == [
        (f'{prefix}/create', {
            'processor_type': 'dataset', 'class_type': 'Dataset', 'init_kwargs': {},
        }, 90),
        (f'{prefix}/create', {
            'processor_type': 'dataloader', 'class_type': 'DataLoader', 'init_kwargs': {'dataset': 'pid:Dataset'},
        }, 90),
        (f'{prefix}/call', {'processor_id': 'pid:Dataset', 'function': 'check', 'call_kwargs': {'flag': True}}, 90),
        (f'{prefix}/call', {
            'processor_id': 'pid:Dataset', 'function': 'encode', 'call_kwargs': {'add_generation_prompt': False},
        }, None),
        (f'{prefix}/call', {'processor_id': 'pid:DataLoader', 'function': 'get_state', 'call_kwargs': {}}, 90),
    ]


def test_client_component_factories_use_the_clients_own_transport() -> None:
    first_session, second_session = _Session(), _Session()
    first = ClientTransport(ClientContext(base_url='http://first', api_key='key'), session=first_session)
    second = ClientTransport(ClientContext(base_url='http://second', api_key='key'), session=second_session)
    first_client, second_client = TwinkleClient(transport=first), TwinkleClient(transport=second)
    set_default_transport(second)
    try:
        packed = first_client.dataset('PackingDataset', None, 4)
        assert isinstance(packed, PackingDataset)
        assert first_session.calls[0][1]['init_kwargs']['packing_num_proc'] == 4
        assert isinstance(first_client.dataset(LazyDataset), LazyDataset)
        assert isinstance(first_client.dataset(), Dataset)
        loader = first_client.dataloader(packed)
        assert loader.get_state() == 'ok'
        with pytest.raises(ValueError, match='same ClientTransport'):
            second_client.dataloader(packed)
        with pytest.raises(ValueError, match='Unknown remote dataset class'):
            first_client.dataset('MissingDataset')
        assert all(url.startswith('http://first/') for url, _, _ in first_session.calls)
        assert not second_session.calls
    finally:
        clear_default_transport(second)
        first_client.close()
        second_client.close()


def test_dataset_meta_data_slice_round_trips() -> None:
    for data_slice in (range(1, 9, 2), [1, 3, 5]):
        restored = deserialize_object(serialize_object(DatasetMeta(dataset_id='demo', data_slice=data_slice)))
        assert restored.dataset_id == 'demo'
        assert list(restored.data_slice) == list(data_slice)


def test_dataset_method_surfaces_and_iterable_mro() -> None:
    assert 'map' not in LazyDataset.__dict__
    assert hasattr(LazyDataset, 'map')
    assert hasattr(PackingDataset, 'map')
    assert '__len__' not in IterableDataset.__dict__
    assert '__getitem__' not in IterableDataset.__dict__
    assert IterableDataset.__mro__[1] is TorchIterableDataset
    assert IterablePackingDataset.__mro__[1] is TorchIterableDataset


def test_dataloader_import_does_not_load_transformers() -> None:
    source = (
        "import json, sys, twinkle_client.dataloader; "
        "print(json.dumps({'transformers': 'transformers' in sys.modules}))"
    )
    result = subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == {'transformers': False}
