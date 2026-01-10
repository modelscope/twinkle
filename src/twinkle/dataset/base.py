# Copyright (c) ModelScope Contributors. All rights reserved.
import os.path
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Type, Union

from datasets import interleave_datasets, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

import twinkle
from twinkle import template, Plugin, preprocessor
from twinkle.hub import HubOperation
from twinkle.infra import remote_class, remote_function
from twinkle.preprocessor import Preprocessor, DataFilter
from twinkle.template import Template
from twinkle.utils.parallel import processing_lock


@dataclass
class DatasetMeta:
    # The dataset id or local path
    dataset_id: str
    # The subset name
    subset_name: str = 'default'
    # The split
    split: str = 'train'
    # Pick a data slice
    data_slice: Iterable = None

    def get_id(self):
        return self.dataset_id.replace(os.sep, '_').replace('.', '_') + ':' + self.subset_name + ':' + self.split


@remote_class(execute='first')
class Dataset(TorchDataset):
    """A dataset wrapper to load and map the dataset.

    Args:
        dataset_meta: A dataset meta information for loading the original dataset.
        kwargs:
            streaming: Whether is streaming mode.
            num_proc: Number of processes to use.
            revision: The revision of the dataset, only available when dataset is id in the hf/ms hub.
            Any other kwargs supported by `datasets.load_dataset`.
    """

    def __init__(self, dataset_meta: DatasetMeta, **kwargs):
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets = {
            dataset_meta.get_id(): dataset
        }
        self.dataset = dataset
        self.template = None

    @remote_function()
    def set_template(self, template_cls: Union[Type[Template], str], **kwargs):
        """Set the template to encode/check the dataset.

        Args:
            template_cls: The template class, or the template plugin, or the template class name to load.
            **kwargs: The template init params.
        """

        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, Template)
        self.template = template_cls(**kwargs)

    @remote_function()
    def encode(self, **kwargs):
        """An inplace operation to encode the dataset.

        Args:
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        with processing_lock('dataset'):
            # use a default lock because encode is to all datasets
            if kwargs.get('batched', True):
                self.dataset = self.dataset.map(self.template.batch_encode, **kwargs).filter(lambda batch: [len(x) > 0 for x in batch['input_ids']], **kwargs)
            else:
                self.dataset = self.dataset.map(self.template.encode, **kwargs).filter(lambda x: len(x['input_ids']) > 0, **kwargs)

    @remote_function()
    def check(self, **kwargs):
        """An inplace operation to check the dataset.

        Args:
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        with processing_lock('dataset'):
            # use a default lock because check is to all datasets
            if kwargs.get('batched', True):
                self.dataset = self.dataset.map(self.template.batch_check, **kwargs).filter(lambda x: x is not None, **kwargs)
            else:
                self.dataset = self.dataset.map(self.template.check, **kwargs).filter(lambda x: x is not None, **kwargs)

    @staticmethod
    def _load_dataset(dataset_meta: DatasetMeta, **kwargs):
        dataset_id = dataset_meta.dataset_id
        subset_name = dataset_meta.subset_name
        split = dataset_meta.split
        with processing_lock(dataset_meta.get_id()):
            if os.path.exists(dataset_id):
                streaming = kwargs.get('streaming', False)
                num_proc = kwargs.get('num_proc', 1)
                ext = os.path.splitext(dataset_id)[1].lstrip('.')
                file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
                kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}
                if file_type == 'csv':
                    kwargs['na_filter'] = False
                dataset = load_dataset(file_type, data_files=dataset_id, **kwargs)
            else:
                dataset = HubOperation.load_dataset(dataset_id, subset_name, split, **kwargs)
        if isinstance(dataset_meta.data_slice, Iterable):
            dataset = dataset.select(dataset_meta.data_slice)
        return dataset

    @remote_function()
    def map(self, preprocess_func: Union[Callable, str, Type[Preprocessor]], dataset_meta: DatasetMeta = None, **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            preprocess_func: A preprocess function, or a `Preprocessor` class name, or a preprocessor plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            **kwargs: The kwargs of the `datasets.map`.
        """

        if isinstance(preprocess_func, Preprocessor):
            preprocess_func = preprocess_func()
        elif isinstance(preprocess_func, str):
            if hasattr(twinkle.preprocessor, preprocess_func):
                preprocess_func = getattr(preprocessor, preprocess_func)()
            else:
                raise ValueError(f'Preprocessor {preprocess_func} not found.')
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        with processing_lock(key):
            self.datasets[key] = self.datasets[key].map(preprocess_func, **kwargs)
        if len(self.datasets) == 1:
            self.dataset = self.datasets[key]

    @remote_function()
    def filter(self, filter_func: Union[Callable, str, Type[DataFilter]], dataset_meta: DatasetMeta = None, **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            filter_func: A filter function, or a `DataFilter` class name, or a filter plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            **kwargs: The kwargs of the `datasets.map`.
        """
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
        with processing_lock(key):
            self.datasets[key] = self.datasets[key].filter(filter_func, **kwargs)
        if len(self.datasets) == 1:
            self.dataset = self.datasets[key]

    @remote_function()
    def add_dataset(self,
                    dataset_meta: DatasetMeta,
                    **kwargs):
        """Add a new dataset.

        Args:
            dataset_meta: The dataset_meta information of the loaded dataset.
        """
        dataset = self._load_dataset(dataset_meta, **kwargs)
        self.datasets[dataset_meta.get_id()] = dataset

    @remote_function()
    def mix_dataset(self, interleave=True):
        """Mix the datasets if `add_dataset` was called.

        Args:
            interleave: Whether to interleave the dataset, or concatenate the dataset.
        """
        if interleave:
            self.dataset = interleave_datasets(list(self.datasets.values()))
        else:
            self.dataset = concatenate_datasets(list(self.datasets.values()))

    @remote_function()
    def __getitem__(self, idx):
        return self.dataset[idx]

    @remote_function()
    def __len__(self):
        return len(self.dataset)
