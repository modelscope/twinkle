from .base import Dataset
from twinkle import remote_function


class IterableDataset(Dataset):
    """An Iterable dataset wrapper."""

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    @remote_function(execute='first')
    def __iter__(self):
        return self.dataset.__iter__()
