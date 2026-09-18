# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Callable, Dict, Optional, Type, Union

from twinkle.dataset import DatasetMeta
from twinkle.preprocessor import DataFilter, Preprocessor
from twinkle.template import Template
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component


class Dataset(object):
    """Client wrapper for Dataset that calls server HTTP endpoints."""

    def __init__(self, dataset_meta: DatasetMeta = None, **kwargs):
        self.processor_id = create_remote_component('dataset', 'Dataset', dataset_meta=dataset_meta, **kwargs)

    def set_template(self, template_func: Union[Template, Type[Template], str], **kwargs):
        return call_remote_component(self.processor_id, 'set_template', template_func=template_func, **kwargs)

    def encode(self, add_generation_prompt: bool = False, timeout: Optional[int] = 600, **kwargs):
        return call_remote_component(
            self.processor_id, 'encode', timeout, add_generation_prompt=add_generation_prompt, **kwargs)

    def check(self, **kwargs):
        return call_remote_component(self.processor_id, 'check', **kwargs)

    def cast_column(self, column: str, decode: bool = True):
        return call_remote_component(self.processor_id, 'cast_column', column=column, decode=decode)

    def map(self,
            preprocess_func: Union[Preprocessor, Callable, str, Type[Preprocessor]],
            dataset_meta: DatasetMeta = None,
            init_args: Dict[str, Any] = None,
            **kwargs):
        return call_remote_component(
            self.processor_id,
            'map',
            preprocess_func=preprocess_func,
            dataset_meta=dataset_meta,
            init_args=init_args,
            **kwargs)

    def filter(self,
               filter_func: Union[Callable, str, Type[DataFilter], DataFilter],
               dataset_meta: DatasetMeta = None,
               init_args: Dict[str, Any] = None,
               **kwargs):
        return call_remote_component(
            self.processor_id,
            'filter',
            filter_func=filter_func,
            dataset_meta=dataset_meta,
            init_args=init_args,
            **kwargs)

    def add_dataset(self, dataset_meta: DatasetMeta, **kwargs):
        return call_remote_component(self.processor_id, 'add_dataset', dataset_meta=dataset_meta, **kwargs)

    def mix_dataset(self, interleave=True):
        return call_remote_component(self.processor_id, 'mix_dataset', interleave=interleave)

    def save_as(self,
                output_path: str,
                format: Optional[str] = None,
                batch_size: int = 1000,
                mode: str = 'immediate',
                **kwargs):
        return call_remote_component(
            self.processor_id,
            'save_as',
            output_path=output_path,
            format=format,
            batch_size=batch_size,
            mode=mode,
            **kwargs)

    def flush_save(self):
        return call_remote_component(self.processor_id, 'flush_save')

    def __getitem__(self, idx):
        return call_remote_component(self.processor_id, '__getitem__', idx=idx)

    def __len__(self):
        return call_remote_component(self.processor_id, '__len__')
