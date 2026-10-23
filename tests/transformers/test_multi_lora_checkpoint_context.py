from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('operation', ['save', 'load'])
def test_checkpoint_activates_target_tenant_before_optimizer_state_initialization(operation, tmp_path):
    manager = MultiLora(max_loras=2, max_r=4)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=['0'])
    model = manager.patch(torch.nn.Sequential(torch.nn.Linear(4, 4)), target_modules=['0'], lora_config=config)
    manager.acquire_lora('first', config)
    manager.acquire_lora('second', config)
    wrapper = object.__new__(MultiLoraTransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.multi_adapter = manager
    wrapper._model_wrapped = True
    wrapper.strategy = SimpleNamespace(unwrap_model=lambda value: value)
    with manager.adapter('first'):
        params = [param for name, param in model.named_parameters() if '.lora_0.' in name]
    optimizer = torch.optim.Adam(params)
    wrapper.optimizer_group = {'first': SimpleNamespace(optimizer=optimizer), 'second': SimpleNamespace()}
    manager.activate_adapter('second')
    assert all(not param.requires_grad for param in params)

    def checkpoint_optimizer(*args, **kwargs):
        state = get_optimizer_state_dict(model, optimizer)
        assert len(state['state']) == len(params)
        assert all('exp_avg' in value for value in state['state'].values())
        assert all(param.requires_grad for param in params)
        return str(tmp_path)

    if operation == 'save':
        with patch.object(TransformersModel, 'save', side_effect=checkpoint_optimizer):
            wrapper.save('checkpoint', adapter_name='first', save_optimizer=True)
    else:
        object.__setattr__(wrapper, '_restore_training_state', checkpoint_optimizer)
        with patch('twinkle.model.transformers.multi_lora_transformers.load_peft_weights', return_value={}), \
                patch.object(manager, 'set_state_dict'):
            wrapper.load(str(tmp_path), adapter_name='first', load_optimizer=True)
    assert model.active_adapter == 'lora_0'
    assert set(model.peft_config) == {'lora_0', 'lora_1'}


def _make_two_tenant_wrapper():
    manager = MultiLora(max_loras=2, max_r=4)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=['0'])
    model = manager.patch(torch.nn.Sequential(torch.nn.Linear(4, 4)), target_modules=['0'], lora_config=config)
    manager.reset_adapter_status()
    manager.acquire_lora('first', config)
    manager.acquire_lora('second', config)
    wrapper = object.__new__(MultiLoraTransformersModel)
    wrapper.multi_adapter = manager
    return wrapper, model


def _lora_params(model):
    return {name: param for name, param in model.named_parameters() if 'lora_' in name}


@pytest.mark.parametrize('disable_lora', [False, True])
def test_adapter_context_keeps_every_slot_trainable(disable_lora):
    wrapper, model = _make_two_tenant_wrapper()
    for tenant in ('second', 'first', 'second'):
        with wrapper._adapter_context(tenant, disable_lora=disable_lora):
            assert all(param.requires_grad for param in _lora_params(model).values())
        assert all(param.requires_grad for param in _lora_params(model).values())


def test_adapter_context_only_active_slot_receives_gradient():
    wrapper, model = _make_two_tenant_wrapper()
    for param in _lora_params(model).values():
        torch.nn.init.normal_(param)
    with wrapper._adapter_context('second'):
        model(torch.randn(2, 4)).sum().backward()
    grads = {name: param.grad for name, param in _lora_params(model).items()}
    assert all(grad is not None for name, grad in grads.items() if '.lora_1.' in name)
    assert all(grad is None for name, grad in grads.items() if '.lora_0.' in name)
