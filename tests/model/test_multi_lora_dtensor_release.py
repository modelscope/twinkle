import pytest
import torch
from peft import LoraConfig
from torch import nn

from twinkle.model.multi_lora import LoraTenant, MultiLora
from twinkle.model.multi_lora_target_parameters import TargetParameterLoraWrapper


class _AutogradView(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):
        return grad


class _FakeDistributedParameter:

    def __init__(self, local_parameter, global_shape):
        self._local_parameter = local_parameter
        self.shape = global_shape
        self.dtype = local_parameter.dtype
        self.device = local_parameter.device
        self.device_mesh = object()
        self.placements = ()

    def to_local(self):
        return _AutogradView.apply(self._local_parameter)


def _make_lora_config(rank):
    return LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=['linear'])


def test_multilora_writes_dtensor_local_autograd_view_without_tracking_gradients():
    local_parameter = nn.Parameter(torch.ones(2))
    distributed_parameter = _FakeDistributedParameter(local_parameter, global_shape=(4,))

    MultiLora()._write_param_tensor(distributed_parameter, torch.zeros(2))

    assert torch.count_nonzero(local_parameter) == 0


def test_target_parameter_lora_writes_dtensor_local_autograd_view_without_tracking_gradients():
    local_parameter = nn.Parameter(torch.ones(2))
    distributed_parameter = _FakeDistributedParameter(local_parameter, global_shape=(4,))

    TargetParameterLoraWrapper._write_parameter(distributed_parameter, torch.zeros(2))

    assert torch.count_nonzero(local_parameter) == 0


def test_multilora_release_keeps_tenant_when_slot_reset_fails(monkeypatch):
    multi_lora = MultiLora(max_loras=1, max_r=4)
    tenant = LoraTenant(
        index=0,
        adapter_name='lora_0',
        config=_make_lora_config(4),
        tenant_adapter_name='adapter_a',
        tenant_config=_make_lora_config(2),
    )
    multi_lora.loras = [tenant]

    def fail_reset(_adapter_name):
        raise RuntimeError('reset failed')

    monkeypatch.setattr(multi_lora, '_load_initial_weights', fail_reset)

    with pytest.raises(RuntimeError, match='reset failed'):
        multi_lora.release_lora('adapter_a')

    assert tenant.tenant_adapter_name == 'adapter_a'
    assert tenant.tenant_config is not None


def test_multilora_release_reports_each_released_slot(monkeypatch):
    multi_lora = MultiLora(max_loras=2, max_r=4)
    slot_config = _make_lora_config(4)
    tenant_config = _make_lora_config(2)
    multi_lora.loras = [
        LoraTenant(
            index=0,
            adapter_name='lora_0',
            config=slot_config,
            tenant_adapter_name='adapter_a',
            tenant_config=tenant_config,
        ),
        LoraTenant(
            index=1,
            adapter_name='lora_1',
            config=slot_config,
            tenant_adapter_name='adapter_b',
            tenant_config=tenant_config,
        ),
    ]
    monkeypatch.setattr(multi_lora, '_load_initial_weights', lambda _adapter_name: None)
    monkeypatch.setattr(multi_lora.target_parameter_manager, 'release', lambda _tenant_name: None)

    assert multi_lora.release_lora('adapter_a') == 'lora_0'
    assert multi_lora._count_available_loras() == 1
    assert multi_lora.release_lora('adapter_b') == 'lora_1'
    assert multi_lora._count_available_loras() == 2
