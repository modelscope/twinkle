# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Dataset
from twinkle import remote_function, remote_class


@remote_class(execute='first')
class IterableDataset(Dataset):
    """An Iterable dataset wrapper."""

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    @remote_function()
    def __iter__(self):
        # TODO if this class passed through actor handler, an error will occur:
        # a global single dataset, multiple dataloaders, the self._iter will cover each other
        return self.dataset.__iter__()
