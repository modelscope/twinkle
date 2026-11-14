# Copyright (c) ModelScope Contributors. All rights reserved.
"""The training loop must hand backward and clipping to a strategy that owns the update.

Accelerate's ``DeepSpeedOptimizerWrapper.step`` and ``.zero_grad`` are ``pass``: the contract is that
the caller drives the engine, which performs the gradient-accumulation accounting, the clipping from
``gradient_clipping`` in the DeepSpeed config, the ZeRO optimizer step and the zeroing. A loop that
calls ``loss.backward()`` and ``optimizer.step()`` instead trains without ever updating a weight --
a 30-step ``sft --deepspeed zero3`` run reported a loss, wrote checkpoints and exited 0 while every
one of the 723 saved tensors stayed bit-identical, and the engine's own counter never advanced, so
every checkpoint was tagged ``global_step0``.

These cases pin the routing itself, which the end-to-end chains can only show after the fact: which
seam the loop takes, what it tells the engine about the accumulation window, whether it builds a
second loss scaler beside the engine's own, and whether it clips a second time over parameters the
engine already clipped and ZeRO already partitioned.
"""
from types import SimpleNamespace

import pytest
import torch

from twinkle.model.transformers import transformers as transformers_module
from twinkle.model.transformers.strategy.base import TransformerStrategyBase
from twinkle.model.transformers.transformers import TransformersModel

backward = TransformersModel.backward.__wrapped__
clip_grad_norm = TransformersModel.clip_grad_norm.__wrapped__


class _Loss:
    """Stands in for the loss tensor, recording whether the loop backwarded it itself."""

    def __init__(self):
        self.backwards = 0

    def backward(self):
        self.backwards += 1


class _Scaler:
    def __init__(self):
        self.scaled = 0

    def scale(self, loss):
        self.scaled += 1
        return loss

    def step(self, optimizer):
        pass

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass

    @staticmethod
    def _found_inf_per_device(optimizer):
        return {}


class _Strategy:
    """Records which seam the loop took."""

    def __init__(self, owns):
        self.owns = owns
        self.calls = []

    def owns_update_step(self):
        return self.owns

    def engine_backward(self, model, loss, *, sync_gradients):
        self.calls.append(('engine_backward', sync_gradients))

    def engine_grad_norm(self, model):
        self.calls.append(('engine_grad_norm',))
        return 1.5

    @staticmethod
    def get_ep_clip_kwargs(model):
        return {}


class _Group:
    def __init__(self, loss, scaler=None):
        self.train_status = SimpleNamespace(loss_value=loss, num_tokens=4)
        self.scaler = scaler
        self.gradient_accumulation_steps = 1
        self.cur_step = 1
        self._last_grad_norm = 0.0
        self._dp_group = None
        self.optimizer = SimpleNamespace(param_groups=[])
        self.lr_scheduler = None

    def do_grad_sync(self, gradient_accumulation_steps=None):
        return True

    def require_finalized(self, operation):
        pass

    def _ensure_dp_group(self):
        pass


class _Loop:
    """Only what backward and clip_grad_norm touch, so the seam is what is under test."""

    def __init__(self, strategy, mixed_precision='bf16', scaler=None):
        self.strategy = strategy
        self.model = object()
        self.mixed_precision = mixed_precision
        self.device_mesh = None
        self.optimizer_group = {'default': _Group(_Loss(), scaler)}
        self.scalers_built = 0

    def _get_default_group(self):
        return 'default'

    def set_grad_scaler(self, adapter_name=None):
        self.scalers_built += 1
        self.optimizer_group['default'].scaler = _Scaler()

    def _get_trainable_parameters(self, adapter_name):
        return {'w': torch.nn.Parameter(torch.zeros(2))}

    @property
    def loss(self):
        return self.optimizer_group['default'].train_status.loss_value


def test_backward_goes_through_the_engine_when_the_strategy_owns_the_update():
    """The defect: on the plain path the update never happens at all."""
    loop = _Loop(_Strategy(owns=True))

    backward(loop)

    assert loop.strategy.calls == [('engine_backward', True)]
    assert loop.loss is None, 'the loss must be released either way'


def test_backward_keeps_the_plain_path_when_the_strategy_does_not_own_the_update():
    loop = _Loop(_Strategy(owns=False))
    loss = loop.loss

    backward(loop)

    assert loss.backwards == 1
    assert loop.strategy.calls == []


def test_backward_tells_the_engine_where_the_accumulation_window_ends():
    """An explicit sync_gradients=False is a non-finalizing backward and must not step."""
    loop = _Loop(_Strategy(owns=True))

    backward(loop, sync_gradients=False)

    assert loop.strategy.calls == [('engine_backward', False)]


def test_no_second_loss_scaler_is_built_beside_the_engine():
    """The engine scales the loss by the window and by its own fp16 factor; a GradScaler doubles it."""
    loop = _Loop(_Strategy(owns=True), mixed_precision='fp16')

    backward(loop)

    assert loop.scalers_built == 0
    assert loop.strategy.calls == [('engine_backward', True)]


def test_a_grad_scaler_is_still_built_on_the_plain_fp16_path():
    loop = _Loop(_Strategy(owns=False), mixed_precision='fp16')

    backward(loop)

    assert loop.scalers_built == 1


def test_an_existing_scaler_is_not_used_when_the_engine_owns_the_update():
    """A scaler built before the strategy was known must not scale the loss a second time."""
    scaler = _Scaler()
    loop = _Loop(_Strategy(owns=True), mixed_precision='fp16', scaler=scaler)

    backward(loop)

    assert scaler.scaled == 0
    assert loop.strategy.calls == [('engine_backward', True)]


def test_clipping_returns_the_engines_norm_instead_of_clipping_again(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('the engine already clipped, over parameters ZeRO already partitioned')

    monkeypatch.setattr(transformers_module, 'normalize_and_clip_grad_norm', refuse)
    loop = _Loop(_Strategy(owns=True))

    assert clip_grad_norm(loop) == 1.5
    assert loop.optimizer_group['default']._last_grad_norm == 1.5
    assert loop.optimizer_group['default'].train_status.num_tokens == 0
    assert ('engine_grad_norm',) in loop.strategy.calls


def test_clipping_still_runs_on_the_plain_path(monkeypatch):
    calls = []

    def fake_clip(parameters, **kwargs):
        calls.append((list(parameters), kwargs.get('max_grad_norm')))
        return 2.5

    monkeypatch.setattr(transformers_module, 'normalize_and_clip_grad_norm', fake_clip)
    monkeypatch.setattr(transformers_module.torch_util, 'gather_object', lambda obj, *a: [obj[0]])
    loop = _Loop(_Strategy(owns=False))

    assert clip_grad_norm(loop, max_grad_norm=0.7) == 2.5
    assert calls and calls[0][1] == 0.7
    assert loop.optimizer_group['default']._last_grad_norm == 2.5


def test_the_base_strategy_does_not_claim_the_update():
    """Only DeepSpeed drives an engine; every other strategy keeps the loop's own step."""
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    from twinkle.model.transformers.strategy.deepspeed import DeepSpeedStrategy
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

    # TransformerStrategyBase is abstract; owns_update_step does not read self, so the unbound
    # function is what declares the default.
    assert TransformerStrategyBase.owns_update_step(None) is False
    for strategy in (AccelerateStrategy, NativeFSDPStrategy):
        assert strategy.owns_update_step(object.__new__(strategy)) is False
    assert DeepSpeedStrategy.owns_update_step(object.__new__(DeepSpeedStrategy)) is True


def test_the_engine_hooks_are_not_silently_inherited(monkeypatch):
    """A strategy that claims the update but lacks the hooks must fail loudly, not train silently."""
    with pytest.raises(NotImplementedError):
        TransformerStrategyBase.engine_backward(object(), None, object(), sync_gradients=True)
    with pytest.raises(NotImplementedError):
        TransformerStrategyBase.engine_grad_norm(object(), None)

