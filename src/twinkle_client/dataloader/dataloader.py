# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type, Union

from twinkle_client.common.remote_component import RemoteComponent
from twinkle_client.http import ClientTransport

if TYPE_CHECKING:
    from twinkle.processor import InputProcessor
    from twinkle_client.dataset import Dataset


class DataLoader(RemoteComponent):
    """Client wrapper for DataLoader that calls server HTTP endpoints."""

    def __init__(
        self,
        dataset: Dataset | Callable,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        dataset_transport = getattr(dataset, '_transport', None)
        if transport is not None and dataset_transport is not None and transport is not dataset_transport:
            raise ValueError('DataLoader and its remote Dataset must use the same ClientTransport')
        self._bind_remote(
            'dataloader', 'DataLoader', dataset=dataset, transport=transport or dataset_transport, **kwargs)

    def __len__(self):
        return self._call('__len__')

    def set_processor(self, processor_cls: type[InputProcessor] | str | InputProcessor | Callable, **kwargs):
        return self._call('set_processor', processor_cls=processor_cls, **kwargs)

    def __iter__(self):
        self._call('__iter__')
        return self

    def __next__(self):
        return self._call('__next__')

    def skip_consumed_samples(self, consumed_train_samples: int):
        return self._call('skip_consumed_samples', consumed_train_samples=consumed_train_samples)

    def resume_from_checkpoint(self, consumed_train_samples, **kwargs):
        return self._call('resume_from_checkpoint', consumed_train_samples=consumed_train_samples, **kwargs)

    def get_state(self):
        return self._call('get_state')
