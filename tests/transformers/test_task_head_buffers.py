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
