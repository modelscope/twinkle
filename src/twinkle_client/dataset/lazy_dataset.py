# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Callable, Dict, Optional, Type, Union

from twinkle.dataset import DatasetMeta
from twinkle.preprocessor import DataFilter, Preprocessor
from twinkle_client.common.component_rpc import create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport
from .base import Dataset


class LazyDataset(Dataset):
    """Client wrapper for LazyDataset that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'dataset', 'LazyDataset', dataset_meta=dataset_meta, transport=self._transport, **kwargs)

    def map(self,
            preprocess_func: Union[Preprocessor, Callable, str, Type[Preprocessor]],
            dataset_meta: DatasetMeta = None,
            init_args: Dict[str, Any] = None,
            **kwargs):
        return self._call(
            'map', preprocess_func=preprocess_func, dataset_meta=dataset_meta, init_args=init_args, **kwargs)

    def filter(self,
               filter_func: Union[Callable, str, Type[DataFilter], DataFilter],
               dataset_meta: DatasetMeta = None,
               init_args: Dict[str, Any] = None,
               **kwargs):
        return self._call('filter', filter_func=filter_func, dataset_meta=dataset_meta, init_args=init_args, **kwargs)

    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        return self._call('add_dataset', dataset_meta=dataset_meta, **kwargs)

    def mix_dataset(self, interleave=True):
        return self._call('mix_dataset', interleave=interleave)

    def encode(self, add_generation_prompt: bool = False, timeout: Optional[int] = 600, **kwargs):
        return self._call('encode', timeout, add_generation_prompt=add_generation_prompt, **kwargs)

    def check(self, **kwargs):
        return self._call('check', **kwargs)

    def __getitem__(self, idx):
        return self._call('__getitem__', idx=idx)

    def __len__(self):
        return self._call('__len__')
