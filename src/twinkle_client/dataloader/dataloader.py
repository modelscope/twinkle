# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Callable, Type, Union

from twinkle.dataset import Dataset
from twinkle.processor import InputProcessor
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component


class DataLoader:
    """Client wrapper for DataLoader that calls server HTTP endpoints."""

    def __init__(self, dataset: Union[Dataset, Callable], **kwargs):
        self.processor_id = create_remote_component('dataloader', 'DataLoader', dataset=dataset, **kwargs)

    def __len__(self):
        return call_remote_component(self.processor_id, '__len__')

    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor, Callable], **kwargs):
        return call_remote_component(self.processor_id, 'set_processor', processor_cls=processor_cls, **kwargs)

    def __iter__(self):
        call_remote_component(self.processor_id, '__iter__')
        return self

    def __next__(self):
        return call_remote_component(self.processor_id, '__next__')

    def skip_consumed_samples(self, consumed_train_samples: int):
        return call_remote_component(
            self.processor_id, 'skip_consumed_samples', consumed_train_samples=consumed_train_samples)

    def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
        return call_remote_component(
            self.processor_id, 'resume_from_checkpoint', consumed_train_samples=consumed_train_samples, **kwargs)

    def get_state(self):
        return call_remote_component(self.processor_id, 'get_state')
