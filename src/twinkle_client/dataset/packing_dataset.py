# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.dataset import DatasetMeta
from twinkle_client.http import ClientTransport
from .base import Dataset


class PackingDataset(Dataset):
    """Client wrapper for PackingDataset that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        packing_num_proc: int = 1,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._bind_remote(
            'dataset',
            'PackingDataset',
            dataset_meta=dataset_meta,
            packing_num_proc=packing_num_proc,
            transport=transport,
            **kwargs,
        )

    def pack_dataset(self):
        return self._call('pack_dataset')

    def __getitem__(self, index):
        return self._call('__getitem__', index=index)

    def __len__(self):
        return self._call('__len__')
