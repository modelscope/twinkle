# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import deepcopy

import torch

from twinkle.model.optimizer_group import BaseOptimizerGroup


def test_accumulated_updates_match_two_complete_batches():
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    reference = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01, foreach=False)
    inputs = torch.randn(8, 3, dtype=torch.float64)
    group = BaseOptimizerGroup(gradient_accumulation_steps=4)
    updates = []
    for index, row in enumerate(inputs):
        (model(row).square().mean() / 4).backward()
        group.cur_step += 1  # TransformersModel.backward increments before consulting the boundary.
        if group.do_grad_sync():
            optimizer.step()
            optimizer.zero_grad()
            updates.append(index + 1)
    for batch in inputs.split(4):
        reference(batch).square().mean().backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()
    assert updates == [4, 8]
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
