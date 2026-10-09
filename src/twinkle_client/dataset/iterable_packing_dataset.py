# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset as TorchIterableDataset
from typing import Type, Union

from twinkle.dataset import DatasetMeta
from twinkle.template import Template
from twinkle_client.common.remote_component import RemoteComponent
from twinkle_client.http import ClientTransport


class IterablePackingDataset(TorchIterableDataset, RemoteComponent):
    """Remote packing iterable backed by one non-reentrant server cursor."""

    def __init__(
        self,
        dataset_meta: DatasetMeta = None,
        packing_interval: int = 128,
        packing_num_proc: int = 1,
        cyclic: bool = False,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._bind_remote(
            'dataset',
            'IterablePackingDataset',
            dataset_meta=dataset_meta,
            packing_interval=packing_interval,
            packing_num_proc=packing_num_proc,
            cyclic=cyclic,
            transport=transport,
            **kwargs,
        )

    def set_template(self, template_cls: Union[Type[Template], str, Template], **kwargs):
        return self._call('set_template', template_cls=template_cls, **kwargs)

    def pack_dataset(self):
        return self._call('pack_dataset')

    def __iter__(self):
        self._call('__iter__')
        return self

    def __next__(self):
        return self._call('__next__')
