# Copyright (c) ModelScope Contributors. All rights reserved.
from torch.utils.data import IterableDataset
from typing import Type, Union

from twinkle.dataset import DatasetMeta
from twinkle.template import Template
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport


class IterablePackingDataset(IterableDataset):
    """Client wrapper for IterablePackingDataset that calls server HTTP endpoints."""

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
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'dataset',
            'IterablePackingDataset',
            dataset_meta=dataset_meta,
            packing_interval=packing_interval,
            packing_num_proc=packing_num_proc,
            cyclic=cyclic,
            transport=self._transport,
            **kwargs,
        )

    def _call(self, function: str, *args, **kwargs):
        return call_remote_component(self.processor_id, function, *args, transport=self._transport, **kwargs)

    def set_template(self, template_cls: Union[Type[Template], str, Template], **kwargs):
        return self._call('set_template', template_cls=template_cls, **kwargs)

    def pack_dataset(self):
        return self._call('pack_dataset')

    def __iter__(self):
        self._call('__iter__')
        return self

    def __next__(self):
        return self._call('__next__')
