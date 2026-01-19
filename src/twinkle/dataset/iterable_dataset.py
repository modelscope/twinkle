# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset

from .base import Dataset, DatasetMeta
from twinkle import remote_function, remote_class


@remote_class(execute='first')
class IterableDataset(IterableDataset, Dataset):
    """An Iterable dataset wrapper."""

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        kwargs['streaming'] = True
        super(IterableDataset, self).__init__(dataset_meta, **kwargs)

    def add_dataset(self,
                    dataset_meta: DatasetMeta,
                    **kwargs):
        kwargs['streaming'] = True
        return super().add_dataset(dataset_meta, **kwargs)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    @remote_function()
    def __iter__(self):
        # TODO if this class passed through actor handler, an error will occur:
        # a global single dataset, multiple dataloaders, the self._iter will cover each other
        return self.dataset.__iter__()
