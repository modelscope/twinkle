# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import json
import subprocess
import sys

from torch.utils.data import IterableDataset as TorchIterableDataset

from twinkle_client.common import remote_component
from twinkle_client.dataloader import DataLoader
from twinkle_client.dataset import Dataset, IterableDataset, IterablePackingDataset, LazyDataset, PackingDataset
from twinkle_client.http import ClientContext, ClientTransport


class _Session:

    def close(self) -> None:
        pass


class _CallableDataset:

    def __call__(self):
        return None


def _transport() -> ClientTransport:
    return ClientTransport(ClientContext(base_url='http://server', api_key='key'), session=_Session())


def test_remote_component_binding_and_dispatch_are_shared(monkeypatch) -> None:
    created: list[tuple[str, str, ClientTransport, dict]] = []
    called: list[tuple[str, str, tuple, ClientTransport, dict]] = []

    def _create(processor_type, class_type, *, transport, **kwargs):
        created.append((processor_type, class_type, transport, kwargs))
        return f'pid:{class_type}'

    def _call(processor_id, function, *args, transport, **kwargs):
        called.append((processor_id, function, args, transport, kwargs))
        return 'result'

    monkeypatch.setattr(remote_component, 'create_remote_component', _create)
    monkeypatch.setattr(remote_component, 'call_remote_component', _call)
    transport = _transport()

    dataset = Dataset(transport=transport)
    loader = DataLoader(_CallableDataset(), transport=transport)

    assert dataset.check(flag=True) == 'result'
    assert loader.get_state() == 'result'
    assert created[0][:3] == ('dataset', 'Dataset', transport)
    assert created[1][:3] == ('dataloader', 'DataLoader', transport)
    assert called[0] == ('pid:Dataset', 'check', (), transport, {'flag': True})
    assert called[1] == ('pid:DataLoader', 'get_state', (), transport, {})


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
