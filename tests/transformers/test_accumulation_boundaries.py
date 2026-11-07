# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import pytest
import torch
from copy import deepcopy
from types import SimpleNamespace

from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('accumulation_steps', [1, 2, 4])
def test_accumulated_updates_match_two_complete_batches(accumulation_steps):
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
    inputs = torch.randn(2 * accumulation_steps, 3, dtype=torch.float64)
    group = BaseOptimizerGroup(gradient_accumulation_steps=accumulation_steps)
    group.optimizer = optimizer
    group.scaler = None
    group.train_status = TrainStatus()
    wrapper = SimpleNamespace(
        model=model,
        optimizer_group={'': group},
        mixed_precision='bf16',
        device_mesh=None,
        _get_default_group=lambda: '')
    updates = []
    handle = optimizer.register_step_post_hook(lambda *args: updates.append(group.cur_step))
    for row in inputs:
        group.train_status.loss_value = model(row).square().mean() / accumulation_steps
        inspect.unwrap(TransformersModel.backward)(wrapper)
        inspect.unwrap(TransformersModel.step)(wrapper)
        inspect.unwrap(TransformersModel.zero_grad)(wrapper)
    handle.remove()
    for batch in inputs.split(accumulation_steps):
        reference(batch).square().mean().backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
    assert updates == [accumulation_steps, 2 * accumulation_steps]
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_counter_continues_to_the_next_complete_boundary():
    group = BaseOptimizerGroup(cur_step=6, gradient_accumulation_steps=4)
    group.cur_step += 1
    assert not group.do_grad_sync()
    group.cur_step += 1
    assert group.do_grad_sync()
    assert group.gradient_accumulation_steps == 4


def test_single_step_accumulation_keeps_every_update():
    group = BaseOptimizerGroup(gradient_accumulation_steps=1)
    for _ in range(3):
        group.cur_step += 1
        assert group.do_grad_sync()
