# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.dataset import DatasetMeta
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from .base import Dataset


class PackingDataset(Dataset):
    """Client wrapper for PackingDataset that calls server HTTP endpoints."""

    def __init__(self, dataset_meta: DatasetMeta = None, packing_num_proc: int = 1, **kwargs):
        self.processor_id = create_remote_component(
            'dataset', 'PackingDataset', dataset_meta=dataset_meta, packing_num_proc=packing_num_proc, **kwargs)

    def pack_dataset(self):
        return call_remote_component(self.processor_id, 'pack_dataset')

    def __getitem__(self, index):
        return call_remote_component(self.processor_id, '__getitem__', index=index)

    def __len__(self):
        return call_remote_component(self.processor_id, '__len__')
