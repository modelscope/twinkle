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


def _slot_module():
    """Resident slots whose names share a prefix, as with max_loras >= 21."""
    root = torch.nn.Module()
    root.layer = torch.nn.Module()
    root.layer.lora_A = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ('lora_2', 'lora_20')})
    root.layer.lora_B = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.ones(2)) for name in ('lora_2', 'lora_21')})
    root.embed = torch.nn.Module()
    root.embed.modules_to_save = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2) for name in ('lora_2', 'lora_29')})
    return root


_EXPECTED_LORA_2 = {
    'layer.lora_A.lora_2.weight', 'layer.lora_A.lora_2.bias', 'layer.lora_B.lora_2',
    'embed.modules_to_save.lora_2.weight', 'embed.modules_to_save.lora_2.bias',
}


def test_transformers_trainable_parameters_do_not_select_prefixed_slots():
    module = _slot_module()
    harness = SimpleNamespace(model=module, strategy=SimpleNamespace(unwrap_model=lambda wrapped: wrapped))
    assert set(TransformersModel._get_trainable_parameters(harness, 'lora_2')) == _EXPECTED_LORA_2


def test_prefixed_slot_does_not_change_optimizer_restore_shape():
    sparse = torch.nn.Module()
    sparse.layer = torch.nn.Module()
    sparse.layer.lora_A = torch.nn.ModuleDict({'lora_2': torch.nn.Linear(2, 2)})
    crowded = _slot_module()
    crowded.layer.lora_A['lora_2'].load_state_dict(sparse.layer.lora_A['lora_2'].state_dict())

    def groups(module):
        harness = SimpleNamespace(model=module, strategy=SimpleNamespace(unwrap_model=lambda wrapped: wrapped))
        params = {name: parameter for name, parameter in
                  TransformersModel._get_trainable_parameters(harness, 'lora_2').items() if '.lora_A.' in name}
        harness._get_trainable_parameters = lambda adapter: params
        return TransformersModel._create_param_group(harness, 'lora_2', weight_decay=0.1)

    source = torch.optim.Adam(groups(sparse), lr=1e-4)
    sparse.layer.lora_A['lora_2'](torch.ones(1, 2)).sum().backward()
    source.step()
    restored = torch.optim.Adam(groups(crowded), lr=1e-4)
    restored.load_state_dict(source.state_dict())
    assert len(restored.state) == len(source.state) == 2


def test_megatron_trainable_parameters_do_not_select_prefixed_slots():
    megatron = pytest.importorskip('twinkle.model.megatron.megatron')
    module = _slot_module()
    harness = SimpleNamespace(model=[module], strategy=SimpleNamespace(unwrap_model=lambda wrapped: wrapped))
    assert set(megatron.MegatronModel._get_trainable_parameters(harness, 'lora_2')) == _EXPECTED_LORA_2
