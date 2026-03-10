# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from collections.abc import Mapping
from typing import Any, List, Union

from twinkle import remote_class, remote_function
from twinkle.data_format import InputFeature, Trajectory
from twinkle.model import MultiLoraTransformersModel


@remote_class()
class TwinkleCompatTransformersModel(MultiLoraTransformersModel):

    @staticmethod
    def _to_cpu_safe_output(obj: Any) -> Any:
        """Convert nested outputs into CPU-safe Python objects for HTTP transport."""
        from twinkle.utils import torch_util

        if isinstance(obj, torch.Tensor):
            tensor = torch_util.to_local_tensor(obj).detach().cpu()
            if tensor.numel() == 1:
                return tensor.item()
            return tensor.tolist()
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Mapping):
            return {key: TwinkleCompatTransformersModel._to_cpu_safe_output(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TwinkleCompatTransformersModel._to_cpu_safe_output(value) for value in obj]
        return obj

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        output = super().forward_backward(inputs=inputs, **kwargs)
        return self._to_cpu_safe_output(output)
