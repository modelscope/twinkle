# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Literal, Optional, Union

from twinkle import DeviceMesh
from twinkle.data_format import InputFeature
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport


class InputProcessor:
    """Client wrapper for InputProcessor that calls server HTTP endpoints."""

    def __init__(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        padding_free: bool = False,
        framework: Literal['transformers', 'megatron'] = 'transformers',
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        self._transport = capture_transport(transport)
        self.processor_id = create_remote_component(
            'processor',
            'InputProcessor',
            device_mesh=device_mesh,
            padding_free=padding_free,
            framework=framework,
            transport=self._transport,
            **kwargs,
        )

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]], **kwargs):
        return call_remote_component(self.processor_id, '__call__', transport=self._transport, inputs=inputs, **kwargs)
