from typing import Type, Union

from twinkle import template, Plugin
from .base import Dataset, DatasetMeta
from .. import remote_function


class LazyDataset(Dataset):

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        super().__init__(dataset_meta, **kwargs)
        self.do_encode = False
        self.do_check = False

    @remote_function(execute='first')
    def encode(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self.do_encode = True

    @remote_function(execute='first')
    def check(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self.do_check = True

    @remote_function(execute='first')
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.do_encode:
            item = self.template.encode(item)
        elif self.do_check:
            item = self.template.check(item)
        return item

    @remote_function(execute='first')
    def __len__(self):
        return len(self.dataset)

