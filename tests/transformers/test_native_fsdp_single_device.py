import numpy as np
import pytest
import torch
from torch import nn

from twinkle import DeviceMesh
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.utils import Platform


class _DeviceTrackingModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.to_devices = []

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device')
        self.to_devices.append(torch.device(device))
        return super().to(*args, **kwargs)


class _FSDPRoot:

    def __init__(self):
        self.events = []

    def unshard(self):
        self.events.append('unshard')

    def reshard(self):
        self.events.append('reshard')


def test_native_fsdp_singleton_mesh_places_model_on_local_device(monkeypatch):
    monkeypatch.setattr(Platform, 'get_local_device', lambda: 'cpu')
    device_mesh = DeviceMesh(
        device_type='cuda',
        mesh=np.array([0]),
        mesh_dim_names=('fsdp', ),
    )
    strategy = NativeFSDPStrategy(
        device_mesh=device_mesh,
        memory_efficient_init=True,
        enable_ep=False,
    )
    model = _DeviceTrackingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original_optimizer_params = list(optimizer.param_groups[0]['params'])

    wrapped_model, wrapped_optimizer = strategy.wrap_model(model, optimizer)

    assert wrapped_model is model
    assert wrapped_optimizer is optimizer
    assert model.to_devices == [torch.device('cpu')]
    assert all(param.device.type == 'cpu' for param in model.parameters())
    assert list(optimizer.param_groups[0]['params']) == original_optimizer_params


def test_native_fsdp_generation_context_unshards_and_restores_root():
    strategy = NativeFSDPStrategy(device_mesh=None, enable_ep=False)
    model = _FSDPRoot()

    with strategy.generation_context(model):
        model.events.append('generate')

    assert model.events == ['unshard', 'generate', 'reshard']


def test_native_fsdp_generation_context_reshards_after_failure():
    strategy = NativeFSDPStrategy(device_mesh=None, enable_ep=False)
    model = _FSDPRoot()

    with pytest.raises(RuntimeError, match='generation failed'):
        with strategy.generation_context(model):
            raise RuntimeError('generation failed')

    assert model.events == ['unshard', 'reshard']
