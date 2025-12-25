import os
from dataclasses import dataclass

from datasets import interleave_datasets, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, Type, Union
from collections.abc import Iterable
from ..hub import MSHub, HFHub
from ..infra import remote_class, remote_function
from ..preprocessor import DataProcessor, DataFilter


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

    def __init__(self, dataset_meta: DatasetMeta, remote_group, **kwargs):
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets = {
            dataset_meta.get_id(): dataset
        }
        self.dataset = dataset

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

    @remote_function(dispatch='all')
    def map(self, preprocess_func: Union[Callable, str, Type[Preprocessor]], dataset_meta: DatasetMeta = None) -> None:
        if isinstance(preprocess_func, DataProcessor):
            preprocess_func = preprocess_func()
        elif isinstance(preprocess_func, str):
            processor_module = Preprocessor.__module__
            if hasattr(processor_module, preprocess_func):
                preprocess_func = getattr(processor_module, preprocess_func)
            else:
                raise ValueError(f'Preprocessor {preprocess_func} not found.')
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        self.datasets[key] = self.datasets[key].map(preprocess_func)

    @remote_function(dispatch='all')
    def filter(self, filter_func: Union[Callable, str, Type[DataFilter]], dataset_meta: DatasetMeta = None) -> None:
        if isinstance(filter_func, DataFilter):
            filter_func = filter_func()
        elif isinstance(filter_func, str):
            processor_module = DataProcessor.__module__
            if hasattr(processor_module, filter_func):
                filter_func = getattr(processor_module, filter_func)
            else:
                raise ValueError(f'Filter {filter_func} not found.')
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        self.datasets[key] = self.datasets[key].filter(filter_func)

    @remote_function(dispatch='all')
    def add_dataset(self,
                    dataset_meta: DatasetMeta,
                    **kwargs):
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets[dataset_meta.get_id()] = dataset

    def mix_dataset(self, interleave=True):
        if interleave:
            self.dataset = interleave_datasets(list(self.datasets.values()))
        else:
            self.dataset = concatenate_datasets(list(self.datasets.values()))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        raise NotImplementedError()
