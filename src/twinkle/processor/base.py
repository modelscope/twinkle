from dataclasses import fields
from typing import Optional, Union, List
import torch
from torch.utils.data import default_collate

from twinkle import DeviceMesh, Platform
from twinkle.data_format import InputFeature


class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None):
        self.device_mesh = device_mesh

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]]):
        if isinstance(inputs, list):
            inputs = self.collate_fn(inputs)
        return self.prepare_inputs(inputs)

    def prepare_inputs(self, inputs: InputFeature) -> InputFeature:
        for f in fields(inputs):
            if isinstance(f.value, torch.Tensor):
                f.value = f.value.to(Platform.get_local_device())
        return inputs

    def collate_fn(self, inputs: List[InputFeature]) -> InputFeature:
        batch_encoded = default_collate([_input.to_transformers_dict() for _input in inputs])
        return InputFeature(**batch_encoded)
