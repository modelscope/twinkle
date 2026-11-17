# Copyright (c) ModelScope Contributors. All rights reserved.
"""Calibration buffers must follow the tenant through reuse and checkpoint restore."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from twinkle.model.base import TrainableModel
from twinkle.model.multi_lora import MultiLoraTaskHeadSlots


def _head():
    head = nn.Linear(2, 2)
    head.register_buffer('temperature', torch.tensor([1., 2.]), persistent=False)
    return head


def _write(tensor, value):
    tensor.copy_(value.to(tensor))


def test_released_head_restores_nonpersistent_calibration():
    slots, host = MultiLoraTaskHeadSlots(2), nn.Module()
    slots.build(host, _head)
    slots.bind('a', host)
    slots.bind('b', host)
    a, b = host.task_heads['a'], host.task_heads['b']
    temperature = a.temperature
    a.temperature.fill_(9)
    slots.release('a', host, _write)
    slots.bind('c', host)
    assert host.task_heads['c'].temperature is temperature
    torch.testing.assert_close(temperature, torch.tensor([1., 2.]))
    torch.testing.assert_close(b.temperature, torch.tensor([1., 2.]))


def test_calibrated_checkpoint_moves_to_another_slot():
    slots, host = MultiLoraTaskHeadSlots(2), nn.Module()
    slots.build(host, _head)
    slots.bind('a', host)
    host.task_heads['a'].temperature.fill_(7)
    saved = slots.export_state('a', host, lambda tensor: tensor)
    slots.release('a', host, _write)
    slots.bind('b', host)
    slots.bind('a', host)
    assert slots.bindings['a'] == 1
    slots.import_state('a', host, saved, _write)
    torch.testing.assert_close(host.task_heads['a'].temperature, torch.full((2,), 7.))
    torch.testing.assert_close(host.task_heads['b'].temperature, torch.tensor([1., 2.]))


def test_single_head_checkpoint_restores_calibration(tmp_path):
    head = _head()
    wrapper = SimpleNamespace(_attached_head=lambda name: head, _task_head_is_master=lambda: True,
                              _task_head_carries_on_this_rank=lambda: True)
    for name in ('_task_head_state', '_restore_task_head_state'):
        setattr(wrapper, name, getattr(TrainableModel, name).__get__(wrapper))
    head.temperature.fill_(5)
    TrainableModel._save_task_head_checkpoint(wrapper, str(tmp_path), None)
    head.temperature.zero_()
    TrainableModel._load_task_head_checkpoint(wrapper, str(tmp_path), None)
    torch.testing.assert_close(head.temperature, torch.full((2,), 5.))


def test_bad_buffer_shape_rejected_before_any_tensor_changes():
    slots, host = MultiLoraTaskHeadSlots(1), nn.Module()
    slots.build(host, _head)
    slots.bind('a', host)
    head = host.task_heads['a']
    before = head.weight.detach().clone()
    state = slots.export_state('a', host, lambda tensor: tensor)
    state['task_head.weight'].fill_(99)
    state['task_head.temperature'] = torch.ones(3)
    with pytest.raises(ValueError, match='shape'):
        slots.import_state('a', host, state, _write)
    torch.testing.assert_close(head.weight, before)


@pytest.fixture(params=[False, True], ids=['plain', 'sharded'])
def head_model(tmp_path, request):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
    from twinkle.model.transformers.transformers import TransformersModel

    sharded = request.param
    if sharded:
        torch.distributed.init_process_group(
            'gloo', init_method=f'file://{tmp_path / "head-pg"}', rank=0, world_size=1)
    try:
        mesh = init_device_mesh('cpu', (1,)) if sharded else None
        head = nn.Linear(2, 1)
        if sharded:
            for name, parameter in list(head.named_parameters()):
                setattr(head, name, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)])))
        host = nn.Module()
        host.task_head = head
        model = object.__new__(TransformersModel)
        model.__dict__['model'] = host
        model.strategy = NativeFSDPStrategy(device_mesh=mesh, enable_ep=False)
        yield model, head
    finally:
        if sharded:
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize('persistent', [False, True])
def test_head_checkpoint_restores_buffers_and_keeps_parameters(head_model, persistent):
    from torch.distributed.tensor import DTensor

    model, head = head_model
    head.register_buffer('calibration', torch.tensor([7.0]), persistent=persistent)
    parameters = dict(head.named_parameters())
    state = model._task_head_state('')
    assert set(state) == {'task_head.weight', 'task_head.bias', 'task_head.calibration'}
    assert all(not isinstance(value, DTensor) and value.device.type == 'cpu' for value in state.values())
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.calibration.zero_()
    model._restore_task_head_state('', state)
    for name, parameter in head.named_parameters():
        assert parameter is parameters[name]
        value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        torch.testing.assert_close(value, state[f'task_head.{name}'])
    torch.testing.assert_close(head.calibration, torch.tensor([7.0]))


@pytest.mark.parametrize('persistent', [False, True])
def test_head_loads_legacy_parameters_without_changing_calibration(head_model, persistent):
    from torch.distributed.tensor import DTensor

    model, head = head_model
    head.register_buffer('calibration', torch.tensor([7.0]), persistent=persistent)
    legacy = {key: torch.full_like(value, 3.0) for key, value in model._task_head_state('').items()
              if key != 'task_head.calibration'}
    model._restore_task_head_state('', legacy)
    for parameter in head.parameters():
        value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        torch.testing.assert_close(value, torch.full_like(value, 3.0))
    torch.testing.assert_close(head.calibration, torch.tensor([7.0]))


def test_head_rejects_bad_buffer_shape_before_loading_parameters(head_model):
    model, head = head_model
    head.register_buffer('calibration', torch.tensor([7.0]), persistent=False)
    before = model._task_head_state('')
    state = {key: value.clone() for key, value in before.items()}
    state['task_head.weight'].fill_(99)
    state['task_head.calibration'] = torch.ones(3)
    with pytest.raises(ValueError, match='shape'):
        model._restore_task_head_state('', state)
    after = model._task_head_state('')
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)
