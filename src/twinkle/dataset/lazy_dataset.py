# Copyright (c) ModelScope Contributors. All rights reserved.

from twinkle import remote_class
from .base import Dataset, DatasetMeta
from .. import remote_function


@remote_class(execute='first')
class LazyDataset(Dataset):
    """A lazy encode dataset wrapper."""

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        super().__init__(dataset_meta, **kwargs)
        self.do_encode = False
        self.do_check = False

    @remote_function()
    def encode(self, **kwargs):
        self.do_encode = True

    @remote_function()
    def check(self, **kwargs):
        self.do_check = True

    @remote_function()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.do_encode:
            item = self.template.encode(item)
        elif self.do_check:
            item = self.template.check(item)
        return item

    @remote_function()
    def __len__(self):
        return len(self.dataset)

