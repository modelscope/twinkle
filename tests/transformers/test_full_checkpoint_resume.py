import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('weights_only', [True, False])
def test_full_resume_restores_weights_and_optional_training_state(tmp_path, weights_only):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    source = torch.nn.Parameter(torch.ones(1, 1))
    saved_optimizer = torch.optim.Adam([source], lr=0.1)
    source.grad = torch.full_like(source, 5)
    saved_optimizer.step()
    torch.save(saved_optimizer.state_dict(), tmp_path / 'optimizer.pt')
    save_file({'weight': torch.full((1, 1), 2.)}, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'trainer_state.json').write_text(json.dumps({
        'cur_step': 3, 'consumed_train_samples': 12, 'gradient_accumulation_steps': 2,
    }))
    wrapper = object.__new__(TransformersModel)
    wrapper.__dict__['model'] = model
    wrapper.strategy = NativeFSDPStrategy(device_mesh=None, enable_ep=False)
    wrapper.optimizer_group = {'': SimpleNamespace(optimizer=optimizer, lr_scheduler=None, scaler=None,
                                                   cur_step=0, gradient_accumulation_steps=1)}
    result = wrapper.resume_from_checkpoint(str(tmp_path), resume_only_model=weights_only)
    assert model(torch.ones(1, 1)).item() == 2
    # dev's weights-only contract starts fresh progress as well as fresh optimizer state.
    assert result['cur_step'] == (0 if weights_only else 3)
    assert result['consumed_train_samples'] == (0 if weights_only else 12)
    if weights_only:
        assert not optimizer.state
        assert wrapper.optimizer_group[''].cur_step == 0
    else:
        torch.testing.assert_close(optimizer.state[model.weight]['exp_avg'], torch.full((1, 1), 0.5))
        assert wrapper.optimizer_group[''].cur_step == 3
        assert wrapper.optimizer_group[''].gradient_accumulation_steps == 2


@pytest.mark.parametrize('offload_model', [False, True])
@pytest.mark.parametrize('offload_optimizer', [False, True])
def test_offload_handle_selects_model_and_optimizer_independently(offload_model, offload_optimizer):
    from unittest.mock import Mock
    from twinkle.model.offload import offload_record

    wrapper = object.__new__(TransformersModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(wrapper.model.parameters())
    wrapper.optimizer_group = {'': SimpleNamespace(optimizer=optimizer)}
    wrapper.strategy = SimpleNamespace(offload_to_cpu=Mock())
    wrapper.offload_to_cpu(offload_model=offload_model, offload_optimizer=offload_optimizer)
    if not offload_model and not offload_optimizer:
        wrapper.strategy.offload_to_cpu.assert_not_called()
        assert offload_record(wrapper.strategy) == (False, False)
    elif offload_model:
        wrapper.strategy.offload_to_cpu.assert_called_once_with(
            wrapper.model, optimizer if offload_optimizer else None)
    else:
        wrapper.strategy.offload_to_cpu.assert_called_once_with(wrapper.model, optimizer, offload_model=False)


@pytest.mark.parametrize('strategy_name', ['accelerate', 'native_fsdp'])
@pytest.mark.parametrize('offload_model', [False, True])
@pytest.mark.parametrize('offload_optimizer', [False, True])
def test_independent_offload_preserves_optimizer_continuation(
        strategy_name, offload_model, offload_optimizer, monkeypatch):
    import copy
    from unittest.mock import Mock
    import twinkle.model.offload as offload
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy

    torch.manual_seed(42)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.ones(2, 2)
    model(inputs).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    if strategy_name == 'accelerate':
        strategy = object.__new__(AccelerateStrategy)
        strategy.accelerator = SimpleNamespace(device=torch.device('cpu'), unwrap_model=lambda module, **kw: module)
    else:
        strategy = NativeFSDPStrategy(device_mesh=None, enable_ep=False)
    model.to = Mock(wraps=model.to)
    move_state = Mock(wraps=offload.move_optimizer_state)
    monkeypatch.setattr(offload, 'move_optimizer_state', move_state)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    strategy.offload_to_cpu(model, optimizer if offload_optimizer else None, offload_model=offload_model)
    assert offload.offload_record(strategy) == (offload_model, offload_optimizer)
    assert model.to.call_count == int(offload_model)
    assert move_state.call_count == int(offload_optimizer)
    strategy.reload_to_gpu(model, optimizer)
    assert offload.offload_record(strategy) == (False, False)
    assert model.to.call_count == 2 * int(offload_model)
    assert move_state.call_count == 2 * int(offload_optimizer)
    strategy.reload_to_gpu(model, optimizer)
    assert model.to.call_count == 2 * int(offload_model)
    assert move_state.call_count == 2 * int(offload_optimizer)
    for module, optim in [(model, optimizer), (reference, reference_optimizer)]:
        module(inputs).sum().backward()
        optim.step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], rtol=0, atol=0)
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, reference_optimizer.state[expected_parameter][key], rtol=0, atol=0)
