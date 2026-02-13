# Copyright (c) ModelScope Contributors. All rights reserved.

from twinkle import remote_class, remote_function
from .base import Dataset, DatasetMeta


@remote_class(execute='first')
class LazyDataset(Dataset):
    """A lazy encode dataset wrapper.

    This class is used to do lazy tokenize to preventing OOM, e.g. multimodal datasets.
    """

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        super().__init__(dataset_meta, **kwargs)
        self.do_encode = False
        self.do_check = False

    @remote_function()
    def encode(self, **kwargs):
        assert self.template is not None
        assert self.template.truncation_strategy != 'split', ('Lazy tokenize does not support '
                                                              'truncation_strategy==`split`')
        self.do_encode = True

    @remote_function()
    def check(self, **kwargs):
        assert self.template is not None
        self.do_check = True

    @remote_function()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # may raise errors
        if self.do_encode:
            item = self.template.batch_encode([item])[0]
        elif self.do_check:
            item = self.template.check(item)
        return item

    @remote_function()
    def __len__(self):
        return len(self.dataset)
