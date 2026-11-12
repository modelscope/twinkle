# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optimizer groups must not depend on DDP wrapping or repeated GA kwargs."""
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.transformers import TransformersModel


def test_forward_kwargs_do_not_duplicate_metric_control_parameters():
    metric = Mock()
    kwargs = {'gradient_accumulation_steps': 2, 'lr': 99, 'step': 99, 'grad_norm': 99, 'loss_reduction': 'sum'}
    status = TrainStatus(inputs=[{}], outputs={'loss': 1.0}, metrics=[metric], forward_kwargs=kwargs)
    group = BaseOptimizerGroup(train_status=status, cur_step=3, gradient_accumulation_steps=2)
    group.accumulate_metrics(True)
    metric.accumulate.assert_called_once_with(
        status.inputs, status.outputs, gradient_accumulation_steps=2, lr=[], step=2, grad_norm=0.0,
        loss_reduction='sum')
    assert kwargs['lr'] == 99


def test_parameter_groups_and_optimizer_restore_are_stable_after_wrapping():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    harness = SimpleNamespace(
        model=model,
        strategy=SimpleNamespace(unwrap_model=lambda wrapped: getattr(wrapped, 'module', wrapped)),
        _get_trainable_parameters=lambda adapter: dict(model.named_parameters()),
    )
    original_groups = TransformersModel._create_param_group(harness, '', weight_decay=0.1)
    optimizer = torch.optim.Adam(original_groups, lr=1e-4)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    harness.model = Wrapper(model)
    late_groups = TransformersModel._create_param_group(harness, '', weight_decay=0.1)
    assert [g['param_names'] for g in late_groups] == [g['param_names'] for g in original_groups]
    assert original_groups[0]['param_names'] == ['0.weight']
    restored = torch.optim.Adam(late_groups, lr=1e-4)
    restored.load_state_dict(state)
    assert len(restored.state) == len(optimizer.state)
