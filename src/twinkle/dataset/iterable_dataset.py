# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset

from twinkle import remote_class, remote_function
from .base import Dataset, DatasetMeta


@remote_class(execute='first')
class IterableDataset(IterableDataset, Dataset):
    """An Iterable dataset wrapper."""

    def __init__(self, dataset_meta: DatasetMeta = None, **kwargs):
        if dataset_meta is not None:
            kwargs['streaming'] = True
        super().__init__(dataset_meta, **kwargs)

    @remote_function()
    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        kwargs['streaming'] = True
        return super().add_dataset(dataset_meta, **kwargs)

    @remote_function()
    def __len__(self):
        raise NotImplementedError()

    @remote_function()
    def __getitem__(self, idx):
        raise NotImplementedError()

    @remote_function()
    def __iter__(self):
        for row in self.dataset:
            self._write_through(row)
            yield row
