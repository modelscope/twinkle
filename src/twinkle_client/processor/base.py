# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Literal, Optional, Union

from twinkle import DeviceMesh
from twinkle.data_format import InputFeature
from twinkle_client.common.component_rpc import call_remote_component, create_remote_component


class InputProcessor(object):
    """Client wrapper for InputProcessor that calls server HTTP endpoints."""

    def __init__(self,
                 device_mesh: Optional[DeviceMesh] = None,
                 padding_free: bool = False,
                 framework: Literal['transformers', 'megatron'] = 'transformers',
                 **kwargs):
        self.processor_id = create_remote_component(
            'processor',
            'InputProcessor',
            device_mesh=device_mesh,
            padding_free=padding_free,
            framework=framework,
            **kwargs)

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]], **kwargs):
        return call_remote_component(self.processor_id, '__call__', inputs=inputs, **kwargs)
