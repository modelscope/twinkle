# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset as TorchIterableDataset

from twinkle.dataset import DatasetMeta
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport


class IterableDataset(TorchIterableDataset):
    """Client wrapper for IterableDataset that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'dataset', 'IterableDataset', dataset_meta=dataset_meta, transport=self._transport, **kwargs)

    def _call(self, function: str, *args, **kwargs):
        return call_remote_component(self.processor_id, function, *args, transport=self._transport, **kwargs)

    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        return self._call('add_dataset', dataset_meta=dataset_meta, **kwargs)

    def __len__(self):
        return self._call('__len__')

    def __getitem__(self, idx):
        return self._call('__getitem__', idx=idx)

    def __iter__(self):
        self._call('__iter__')
        return self

    def __next__(self):
        return self._call('__next__')
