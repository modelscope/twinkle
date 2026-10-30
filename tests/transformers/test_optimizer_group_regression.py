# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optimizer groups must not depend on DDP wrapping or repeated GA kwargs."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.transformers import OptimizerGroup, TransformersModel


@pytest.mark.parametrize('group_cls', [BaseOptimizerGroup, OptimizerGroup])
@pytest.mark.parametrize('is_training', [True, False])
def test_forward_kwargs_do_not_duplicate_metric_control_parameters(group_cls, is_training):
    metric = Mock(_twinkle_loss_key=None)
    kwargs = {'gradient_accumulation_steps': 99, 'lr': 99, 'step': 99, 'grad_norm': 99,
              'loss_reduction': 'mean', 'custom_metric_arg': 7}
    original_kwargs = dict(kwargs)
    status = TrainStatus(inputs=[{}], outputs={'loss': 1.0}, metrics=[metric], forward_kwargs=kwargs,
                         loss_reduction='sum')
    group = group_cls(cur_step=3, gradient_accumulation_steps=2, _last_grad_norm=1.25)
    if is_training:
        group.train_status = status
    else:
        group.eval_status = status
    group.accumulate_metrics(is_training)
    metric.accumulate.assert_called_once_with(
        status.inputs, status.outputs, gradient_accumulation_steps=2, lr=[], step=2, grad_norm=1.25,
        loss_reduction='sum', custom_metric_arg=7)
    assert kwargs == original_kwargs


@pytest.mark.parametrize('group_cls', [BaseOptimizerGroup, OptimizerGroup])
@pytest.mark.parametrize('loss_reduction', [None, 'sum'])
def test_shared_metric_arguments_preserve_keyed_loss_dispatch(group_cls, loss_reduction):
    selected, unrelated, default = [Mock(_twinkle_loss_key=key) for key in ['selected', 'other', None]]
    kwargs = {'_metric_loss_key': 'selected', 'step': 99, 'loss_reduction': 'none'}
    status = TrainStatus(inputs=[{}], outputs={'loss': 1.0}, metrics=[selected, unrelated, default],
                         forward_kwargs=kwargs, loss_reduction=loss_reduction)
    group = group_cls(cur_step=3)
    group.set_loss(SimpleNamespace(reduction='mean'), loss_key='selected')
    group.train_status = status
    group.accumulate_metrics(True, include_default=False, metric_step=17)
    selected.accumulate.assert_called_once_with(
        status.inputs, status.outputs, step=17, loss_reduction=loss_reduction or 'mean',
        lr=[], gradient_accumulation_steps=1, grad_norm=0.0)
    unrelated.accumulate.assert_not_called()
    default.accumulate.assert_not_called()
    assert kwargs == {'_metric_loss_key': 'selected', 'step': 99, 'loss_reduction': 'none'}


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
