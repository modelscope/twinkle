import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Type, Union

from datasets import interleave_datasets, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset

import twinkle
from twinkle import template, Plugin
from twinkle.hub import MSHub, HFHub
from twinkle.infra import remote_class, remote_function
from twinkle.preprocessor import Preprocessor, DataFilter


@dataclass
class DatasetMeta:

    dataset_id: str
    subset_name: str = 'default'
    split: str = 'train'
    data_slice: Union[slice, Iterable] = slice(None)

    def get_id(self):
        return self.dataset_id + ':' + self.subset_name + ':' + self.split


@remote_class()
class Dataset(TorchDataset):

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets = {
            dataset_meta.get_id(): dataset
        }
        self.dataset = dataset
        self.template = None

    @remote_function(execute='first')
    def set_template(self, template_cls: Union[Type[template.Template], str], model_id: str, **template_params):
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        self.template = template_cls(model_id, **template_params)

    @remote_function(execute='first')
    def encode(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self.dataset = self.dataset.map(self.template.encode, **kwargs).filter(lambda x: x is not None, **kwargs)

    @remote_function(execute='first')
    def check(self, template_cls: Union[Type[template.Template], str], **kwargs):
        self.dataset = self.dataset.map(self.template.check, **kwargs).filter(lambda x: x is not None, **kwargs)

    @staticmethod
    def _load_dataset(dataset_meta: DatasetMeta, **kwargs):
        dataset_id = dataset_meta.dataset_id
        subset_name = dataset_meta.subset_name
        split = dataset_meta.split
        if dataset_id.startswith('hf://'):
            dataset = HFHub.load_dataset(dataset_id[len('hf://'):], subset_name, split, **kwargs)
        elif dataset_id.startswith('ms://'):
            dataset = MSHub.load_dataset(dataset_id, subset_name, split, **kwargs)
        elif os.path.exists(dataset_id):
            dataset = MSHub.load_dataset(dataset_id, subset_name, split, **kwargs)
        else:
            raise FileNotFoundError(f'Dataset {dataset_id} not found.')

        if isinstance(dataset_meta.data_slice, slice):
            dataset = dataset[dataset_meta.data_slice]
        elif isinstance(dataset_meta.data_slice, Iterable):
            dataset = dataset.select(dataset_meta.data_slice)
        return dataset

    @remote_function(execute='first')
    def map(self, preprocess_func: Union[Callable, str, Type[Preprocessor]], dataset_meta: DatasetMeta = None, **kwargs) -> None:
        if isinstance(preprocess_func, Preprocessor):
            preprocess_func = preprocess_func()
        elif isinstance(preprocess_func, str):
            if hasattr(twinkle.preprocessor, preprocess_func):
                preprocess_func = getattr(twinkle.preprocessor, preprocess_func)()
            else:
                raise ValueError(f'Preprocessor {preprocess_func} not found.')
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        self.datasets[key] = self.datasets[key].map(preprocess_func, **kwargs)

    @remote_function(execute='first')
    def filter(self, filter_func: Union[Callable, str, Type[DataFilter]], dataset_meta: DatasetMeta = None, **kwargs) -> None:
        if isinstance(filter_func, DataFilter):
            filter_func = filter_func()
        elif isinstance(filter_func, str):
            if hasattr(twinkle.preprocessor, filter_func):
                filter_func = getattr(twinkle.preprocessor, filter_func)
            else:
                raise ValueError(f'Filter {filter_func} not found.')
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        self.datasets[key] = self.datasets[key].filter(filter_func, **kwargs)

    @remote_function(execute='first')
    def add_dataset(self,
                    dataset_meta: DatasetMeta,
                    **kwargs):
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets[dataset_meta.get_id()] = dataset

    @remote_function(execute='first')
    def mix_dataset(self, interleave=True):
        if interleave:
            self.dataset = interleave_datasets(list(self.datasets.values()))
        else:
            self.dataset = concatenate_datasets(list(self.datasets.values()))

    @remote_function(execute='first')
    def __getitem__(self, idx):
        return self.dataset[idx]

    @remote_function(execute='first')
    def __len__(self):
        return len(self.dataset)
