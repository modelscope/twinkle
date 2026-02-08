# Copyright (c) ModelScope Contributors. All rights reserved.
import os.path
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Callable, Type, Union, Dict, Any

from datasets import interleave_datasets, concatenate_datasets, load_dataset, IterableDataset
from torch.utils.data import Dataset as TorchDataset

import twinkle
from twinkle import preprocessor
from twinkle.hub import HubOperation
from twinkle.infra import remote_class, remote_function
from twinkle.preprocessor import Preprocessor, DataFilter
from twinkle.template import Template
from twinkle.utils import construct_class
from twinkle.utils import processing_lock


@dataclass
class DatasetMeta:
    """
    The dataset meta-information, used to describe a dataset.
    """
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
    
    def __post_init__(self):
        if self.data_slice is not None and not isinstance(self.data_slice, Iterable):
            raise ValueError('data_slice must be an iterable')


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
    def set_template(self, template_func: Union[Template, Type[Template], str], **kwargs):
        """Set the template to encode/check the dataset.

        Args:
            template_func: The template class/instance, or the template plugin, or the template class name to load.
            **kwargs: The template init params.
        """
        self.template = construct_class(template_func, Template, twinkle.template, **kwargs)

    @remote_function()
    def encode(self, add_generation_prompt: bool = False, **kwargs):
        """An inplace operation to encode the dataset.

        Args:
            add_generation_prompt: If True, append generation prompt suffix
                (e.g. ``<|im_start|>assistant\\n``) to each encoded sample.
                Useful when the encoded dataset will be used for sampling/inference.
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        kwargs['batched'] = True # Only supported batched, because a single row may explode to several rows
        if 'load_from_cache_file' not in kwargs:
            # By default, we don't use load_from_cache_file, because read cache will not consider the changes in the same file,
            # which will cause unexpected behaviors.
            kwargs['load_from_cache_file'] = False
        from functools import partial
        encode_fn = partial(self.template.batch_encode, add_generation_prompt=add_generation_prompt)
        with processing_lock('dataset'):
            # use a default lock because encode is to all datasets
            self.dataset = self.dataset.map(encode_fn, **kwargs).filter(lambda batch: [len(x) > 0 for x in batch['input_ids']], **kwargs)

    @remote_function()
    def check(self, **kwargs):
        """An inplace operation to check the dataset.

        Args:
            **kwargs: The mapping and filter kwargs of the `datasets.map`.
        """
        kwargs['batched'] = True  # Only supported batched, because a single row may explode to several rows
        # check depends on template/tokenizer behavior; cached filter results can keep old empty outputs.
        # Disable cache here to avoid the "silent stop" caused by stale empty cache.
        kwargs.setdefault('load_from_cache_file', False)
        with processing_lock('dataset'):
            # use a default lock because check is to all datasets
            def _check_batch(batch):
                # HF datasets.map expects dict/None; filter expects bool mask, so adapt batch_check output.
                rows = self.template.map_col_to_row(batch) if isinstance(batch, Mapping) else batch
                checked = self.template.batch_check(rows)
                return [item is not None for item in checked]

            self.dataset = self.dataset.filter(_check_batch, **kwargs)

    @staticmethod
    def _load_dataset(dataset_meta: DatasetMeta, **kwargs):
        dataset_id = dataset_meta.dataset_id
        subset_name = dataset_meta.subset_name
        split = dataset_meta.split
        with processing_lock(dataset_meta.get_id()):
            if os.path.exists(dataset_id):
                ext = os.path.splitext(dataset_id)[1].lstrip('.')
                file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
                if file_type == 'csv':
                    kwargs['na_filter'] = False
                dataset = load_dataset(file_type, data_files=dataset_id, **kwargs)
            else:
                dataset = HubOperation.load_dataset(dataset_id, subset_name, split, **kwargs)
        if isinstance(dataset_meta.data_slice, Iterable) and hasattr(dataset, '__len__'):
            
            iter_list = []
            _data_len = len(dataset)
            for idx in dataset_meta.data_slice:
                if idx >= _data_len:
                    # Prevent out of range, repeat sampling
                    idx = idx % _data_len
                iter_list.append(idx)

            dataset = dataset.select(iter_list)
        return dataset

    @remote_function()
    def map(self, preprocess_func: Union[Preprocessor, Callable, str, Type[Preprocessor]],
            dataset_meta: DatasetMeta = None, init_args: Dict[str, Any] = None, **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            preprocess_func: A preprocess function, or a `Preprocessor` class/instance, or a preprocessor plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            init_args: The init args to construct the preprocessor.
            **kwargs: The kwargs of the `datasets.map`.
        """
        init_args = init_args or {}
        if 'load_from_cache_file' not in kwargs:
            # By default, we don't use load_from_cache_file, because read cache will not consider the changes in the same file,
            # which will cause unexpected behaviors.
            kwargs['load_from_cache_file'] = False
        preprocess_func = construct_class(preprocess_func, Preprocessor, twinkle.preprocessor, **init_args)
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        kwargs['batched'] = False # TODO temporary change to False, because the interface does not support batched
        with processing_lock(key):
            self.datasets[key] = self.datasets[key].map(preprocess_func, **kwargs).filter(lambda x: x is not None, **kwargs) # filter none rows
        if len(self.datasets) == 1:
            self.dataset = self.datasets[key]

    @remote_function()
    def filter(self, filter_func: Union[Callable, str, Type[DataFilter], DataFilter], dataset_meta: DatasetMeta = None,
               init_args: Dict[str, Any] = None,
               **kwargs) -> None:
        """An inplace method to operate or transform the dataset.

        Args:
            filter_func: A filter function, or a `DataFilter` class name, or a filter plugin name.
            dataset_meta: The dataset_meta information of the loaded dataset.
            init_args: The init args to construct the filter.
            **kwargs: The kwargs of the `datasets.map`.
        """
        init_args = init_args or {}
        filter_func = construct_class(filter_func, DataFilter, twinkle.preprocessor, **init_args)
        if dataset_meta is None:
            assert len(self.datasets) == 1
            key = next(iter(self.datasets.keys()))
        else:
            key = dataset_meta.get_id()
        kwargs['batched'] = False  # TODO temporary change to False, because the interface does not support batched
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
        if len(self.datasets) > 1:
            dataset_types = [isinstance(ds, IterableDataset) for ds in self.datasets]
            assert all(dataset_types) or not any(dataset_types), 'All datasets must be all streaming=True or streaming=False'
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
