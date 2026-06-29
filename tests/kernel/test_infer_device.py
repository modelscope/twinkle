import torch
import torch.nn as nn

from twinkle.kernel.core import _infer_device


class _NoParamsNoBuffers(nn.Module):
    pass


class _OnlyBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.zeros(2))


def test_infer_device_from_parameter():
    m = nn.Linear(2, 3)
    assert _infer_device(m) == 'cpu'


def test_infer_device_from_buffer_when_no_params():
    m = _OnlyBuffer()
    assert _infer_device(m) == 'cpu'


def test_infer_device_defaults_to_cpu_when_empty():
    m = _NoParamsNoBuffers()
    assert _infer_device(m) == 'cpu'