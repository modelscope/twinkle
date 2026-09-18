# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Callable, Dict, Optional, Type, Union

from twinkle.dataset import DatasetMeta
from twinkle.preprocessor import DataFilter, Preprocessor
from twinkle.template import Template
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport


class Dataset:
    """Client wrapper for Dataset that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'dataset', 'Dataset', dataset_meta=dataset_meta, transport=self._transport, **kwargs)

    def _call(self, function: str, *args, **kwargs):
        return call_remote_component(self.processor_id, function, *args, transport=self._transport, **kwargs)

    def set_template(self, template_func: Union[Template, Type[Template], str], **kwargs):
        return self._call('set_template', template_func=template_func, **kwargs)

    def encode(self, add_generation_prompt: bool = False, timeout: Optional[int] = 600, **kwargs):
        return self._call('encode', timeout, add_generation_prompt=add_generation_prompt, **kwargs)

    def check(self, **kwargs):
        return self._call('check', **kwargs)

    def cast_column(self, column: str, decode: bool = True):
        return self._call('cast_column', column=column, decode=decode)

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

    def save_as(self,
                output_path: str,
                format: Optional[str] = None,
                batch_size: int = 1000,
                mode: str = 'immediate',
                **kwargs):
        return self._call('save_as', output_path=output_path, format=format, batch_size=batch_size, mode=mode, **kwargs)

    def flush_save(self):
        return self._call('flush_save')

    def __getitem__(self, idx):
        return self._call('__getitem__', idx=idx)

    def __len__(self):
        return self._call('__len__')
