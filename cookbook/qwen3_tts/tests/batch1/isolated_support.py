"""Explicit CPU test doubles for Twinkle infrastructure only.

Production tensor operations and transformers loss remain real. This is NOT
Twinkle/NPU integration evidence. Only loaded with --isolated-twinkle.
"""
import sys
import types
from pathlib import Path
import torch


def install(root):
    def module(name, **attrs):
        obj = types.ModuleType(name)
        obj.__dict__.update(attrs)
        sys.modules[name] = obj
        return obj

    for name, path in [('tts', 'tts'), ('tts.qwen3', 'tts/qwen3')]:
        module(name, __path__=[str(root / path)])

    class Processor:
        padding_map = {}
        VLM_CONCAT_FIELDS = set()
        def __init__(self, **kwargs):
            self.padding_free = kwargs.get('padding_free', False)

    def pad_stack(values, pad_value, concat):
        assert concat is False
        return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=pad_value)

    module('twinkle', remote_function=lambda **kw: lambda fn: fn,
           Platform=types.SimpleNamespace(get_local_device=lambda: 'cpu'))
    module('twinkle.data_format', InputFeature=dict, Trajectory=dict, LossOutput=dict)
    module('twinkle.data_format.output', LossOutput=dict)
    module('twinkle.processor')
    module('twinkle.processor.base', InputProcessor=Processor)
    module('twinkle.utils')
    module('twinkle.utils.torch_utils', pad_and_stack_tensors=pad_stack)
    module('twinkle.loss')
    module('twinkle.loss.base', Loss=object)
    module('twinkle.infra', collect_tensor_dict=lambda x: x)
    module('twinkle.model')
    module('twinkle.model.transformers', TransformersModel=object)
