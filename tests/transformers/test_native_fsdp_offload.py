# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU placement probes for the lazy-wrap / colocated-offload boundary."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from twinkle.model import offload
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy


class PlacementModule:
    """Track placement without allocating CUDA tensors or claiming to run FSDP."""

    def __init__(self, device='cpu'):
        self.param = SimpleNamespace(device=torch.device(device), grad=None)
        self.inv_freq = SimpleNamespace(device=torch.device(device))
        self.moves = []

    def parameters(self):
        yield self.param

    def to(self, device, **kwargs):
        self.param.device = self.inv_freq.device = torch.device(device)
        self.moves.append(self.param.device)
        return self


@pytest.fixture
def strategy(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(offload, 'release_caching_allocator', lambda: None)
    return NativeFSDPStrategy(enable_ep=False)


@pytest.mark.parametrize('offload_model,with_optimizer', [(True, False), (False, True), (True, True)])
def test_reload_uses_each_cycles_device_after_lazy_wrap(strategy, monkeypatch, offload_model, with_optimizer):
    module = PlacementModule()
    optimizer = object() if with_optimizer else None
    move_state = Mock()
    monkeypatch.setattr(offload, 'move_optimizer_state', move_state)

    # The initial CPU placement is legitimate; wrapping has not happened yet.
    strategy.offload_to_cpu(module, optimizer, offload_model=offload_model)
    strategy.reload_to_gpu(module, optimizer)
    assert module.param.device == module.inv_freq.device == torch.device('cpu')

    # Simulate lazy wrapping, then two further cycles (including a changed rank device).
    for device in (torch.device('cuda:0'), torch.device('cuda:1')):
        module.to(device)
        strategy.offload_to_cpu(module, optimizer, offload_model=offload_model)
        expected = torch.device('cpu') if offload_model else device
        assert module.param.device == module.inv_freq.device == expected
        strategy.reload_to_gpu(module, optimizer)
        assert module.param.device == module.inv_freq.device == device
        if with_optimizer:
            assert move_state.call_args.args == (optimizer, device)
        assert offload.offload_record(strategy) == (False, False)


@pytest.mark.parametrize('first_model,first_optimizer', [(True, False), (False, True), (True, True)])
def test_repeated_offload_keeps_restore_target_and_pending_pieces(strategy, monkeypatch, first_model, first_optimizer):
    module = PlacementModule('cuda:1')
    optimizer = object()
    move_state = Mock()
    monkeypatch.setattr(offload, 'move_optimizer_state', move_state)

    strategy.offload_to_cpu(module, optimizer if first_optimizer else None, offload_model=first_model)
    strategy.offload_to_cpu(module, optimizer)
    # A narrower repeated request must not forget the optimizer or capture CPU as the target.
    strategy.offload_to_cpu(module)
    strategy.offload_to_cpu(module, offload_model=False)
    strategy.reload_to_gpu(module, optimizer)
    assert module.param.device == module.inv_freq.device == torch.device('cuda:1')
    assert move_state.call_args.args == (optimizer, torch.device('cuda:1'))
    assert offload.offload_record(strategy) == (False, False)

    moves = list(module.moves)
    calls = move_state.call_count
    strategy.reload_to_gpu(module, optimizer)
    assert module.moves == moves
    assert move_state.call_count == calls


def test_failed_reload_can_be_retried(strategy, monkeypatch):
    module = PlacementModule('cuda:0')
    strategy.offload_to_cpu(module)
    original_to = module.to
    monkeypatch.setattr(module, 'to', Mock(side_effect=RuntimeError('copy failed')))
    with pytest.raises(RuntimeError, match='copy failed'):
        strategy.reload_to_gpu(module)
    assert offload.offload_record(strategy) == (True, False)
    monkeypatch.setattr(module, 'to', original_to)
    strategy.reload_to_gpu(module)
    assert module.param.device == module.inv_freq.device == torch.device('cuda:0')


def test_failed_synchronize_keeps_pending_reload(strategy, monkeypatch):
    module = PlacementModule('cuda:0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    synchronize = Mock(side_effect=[RuntimeError('synchronize failed'), None])
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    strategy.offload_to_cpu(module)
    with pytest.raises(RuntimeError, match='synchronize failed'):
        strategy.reload_to_gpu(module)
    assert offload.offload_record(strategy) == (True, False)
    strategy.reload_to_gpu(module)
    assert offload.offload_record(strategy) == (False, False)


def test_cpu_roundtrip_preserves_real_nonpersistent_buffer_gradients_and_adam(strategy):
    module = torch.nn.Linear(2, 2)
    module.register_buffer('inv_freq', torch.ones(2), persistent=False)
    optimizer = torch.optim.AdamW(module.parameters(), foreach=False)
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    before_params = [param.detach().clone() for param in module.parameters()]
    before_grads = [param.grad.clone() for param in module.parameters()]
    before_buffer = module.inv_freq.clone()
    before_state = {param: {name: value.clone() for name, value in state.items()}
                    for param, state in optimizer.state.items()}

    strategy.offload_to_cpu(module, optimizer)
    strategy.reload_to_gpu(module, optimizer)

    assert 'inv_freq' not in module.state_dict()
    torch.testing.assert_close(module.inv_freq, before_buffer)
    for param, before, grad in zip(module.parameters(), before_params, before_grads):
        torch.testing.assert_close(param, before)
        torch.testing.assert_close(param.grad, grad)
        for name, value in optimizer.state[param].items():
            torch.testing.assert_close(value, before_state[param][name])
    optimizer.zero_grad()
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert any(not torch.equal(param, before) for param, before in zip(module.parameters(), before_params))
