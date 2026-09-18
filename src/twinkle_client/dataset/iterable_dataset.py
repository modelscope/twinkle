# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset as TorchIterableDataset

from twinkle.dataset import DatasetMeta
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component


class IterableDataset(TorchIterableDataset):
    """Client wrapper for IterableDataset that calls server HTTP endpoints."""

    def __init__(self, dataset_meta: DatasetMeta = None, **kwargs):
        self.processor_id = create_remote_component('dataset', 'IterableDataset', dataset_meta=dataset_meta, **kwargs)

    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        return call_remote_component(self.processor_id, 'add_dataset', dataset_meta=dataset_meta, **kwargs)

    def __len__(self):
        return call_remote_component(self.processor_id, '__len__')

    def __getitem__(self, idx):
        return call_remote_component(self.processor_id, '__getitem__', idx=idx)

    def __iter__(self):
        call_remote_component(self.processor_id, '__iter__')
        return self

    def __next__(self):
        return call_remote_component(self.processor_id, '__next__')
