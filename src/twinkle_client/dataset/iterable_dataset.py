# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset as TorchIterableDataset

from twinkle.dataset import DatasetMeta
from twinkle_client.common.remote_component import RemoteComponent
from twinkle_client.http import ClientTransport


class IterableDataset(TorchIterableDataset, RemoteComponent):
    """Remote iterable backed by one server-side cursor.

    Iteration is stateful and does not support concurrent or repeated iteration
    over the same wrapper instance.
    """

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._bind_remote('dataset', 'IterableDataset', dataset_meta=dataset_meta, transport=transport, **kwargs)

    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        return self._call('add_dataset', dataset_meta=dataset_meta, **kwargs)

    def __iter__(self):
        self._call('__iter__')
        return self

    def __next__(self):
        return self._call('__next__')
