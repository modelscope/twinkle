# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.dataset import DatasetMeta
from twinkle_client.common.component_rpc import create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport
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
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'dataset',
            'PackingDataset',
            dataset_meta=dataset_meta,
            packing_num_proc=packing_num_proc,
            transport=self._transport,
            **kwargs,
        )

    def pack_dataset(self):
        return self._call('pack_dataset')

    def __getitem__(self, index):
        return self._call('__getitem__', index=index)

    def __len__(self):
        return self._call('__len__')
