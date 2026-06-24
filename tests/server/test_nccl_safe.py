# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for NCCL-safe fault tolerance utilities.

Organized in five tiers:
1. Unit: _is_fail_fast(), safe_loss(), _zero_loss()
2. Unit: @nccl_safe decorator
3. Unit: BaseOptimizerGroup.__setattr__ auto-wrapping
4. Integration: real loss functions (GRPO, CrossEntropy) through safe_loss
5. Adversarial: malicious data injection, monkey-patching, stability
"""
import pytest
import torch

from unittest.mock import MagicMock

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss
from twinkle.loss.cross_entropy import CrossEntropyLoss
from twinkle.loss.grpo import GRPOLoss
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.utils.nccl_safe import (
    _is_fail_fast,
    _zero_loss,
    nccl_safe,
    safe_loss,
)


# ─── Helpers ──────────────────────────────────────────────────────────────


class DummyLoss(Loss):
    """Simple loss returning sum of logps."""
    require_logps = True
    require_entropy = False
    require_logits = False

    def __call__(self, inputs, outputs, **kwargs):
        logps = outputs['logps']
        return LossOutput(loss=logps.sum(), num_tokens=logps.numel())


class ExplodingLoss(Loss):
    """Loss that always raises RuntimeError."""
    require_logps = True
    require_entropy = True
    require_logits = False

    def __call__(self, inputs, outputs, **kwargs):
        raise RuntimeError('Simulated loss explosion')


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _production_mode(monkeypatch):
    """Set TWINKLE_FAIL_FAST=0 for all tests (production mode)."""
    monkeypatch.setenv('TWINKLE_FAIL_FAST', '0')


@pytest.fixture
def _dev_mode(monkeypatch):
    """Switch to development mode (TWINKLE_FAIL_FAST=1)."""
    monkeypatch.setenv('TWINKLE_FAIL_FAST', '1')


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit Tests: _is_fail_fast
# ═══════════════════════════════════════════════════════════════════════════


class TestIsFailFast:

    def test_default_is_fail_fast(self, monkeypatch):
        monkeypatch.delenv('TWINKLE_FAIL_FAST', raising=False)
        assert _is_fail_fast() is True

    def test_explicit_1(self, monkeypatch):
        monkeypatch.setenv('TWINKLE_FAIL_FAST', '1')
        assert _is_fail_fast() is True

    def test_explicit_0(self, monkeypatch):
        monkeypatch.setenv('TWINKLE_FAIL_FAST', '0')
        assert _is_fail_fast() is False

    @pytest.mark.parametrize('val', ['no', 'false', 'off', 'NO', 'False', 'OFF'])
    def test_falsy_strings(self, monkeypatch, val):
        monkeypatch.setenv('TWINKLE_FAIL_FAST', val)
        assert _is_fail_fast() is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Unit Tests: safe_loss
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeLoss:

    def test_transparent_in_dev_mode(self, _dev_mode):
        """In dev mode, safe_loss still wraps but wrapper propagates exceptions."""
        loss = ExplodingLoss()
        wrapped = safe_loss(loss)
        assert wrapped is not loss  # always wrapped now
        assert wrapped._nccl_safe_wrapped is True
        with pytest.raises(RuntimeError, match='Simulated'):
            wrapped({}, {'logps': torch.tensor([1.0])})

    def test_wraps_in_production(self):
        loss = DummyLoss()
        wrapped = safe_loss(loss)
        assert wrapped is not loss
        assert callable(wrapped)

    def test_idempotent(self):
        loss = DummyLoss()
        w1 = safe_loss(loss)
        w2 = safe_loss(w1)
        assert w1 is w2

    def test_forwards_attributes(self):
        loss = DummyLoss()
        w = safe_loss(loss)
        assert w.require_logps is True
        assert w.require_entropy is False
        assert w.require_logits is False
        assert w._nccl_safe_wrapped is True

    def test_preserves_custom_entropy_flag(self):
        loss = GRPOLoss(entropy_coef=0.1)
        assert loss.require_entropy is True
        w = safe_loss(loss)
        assert w.require_entropy is True

    def test_normal_call_passes_through(self):
        w = safe_loss(DummyLoss())
        result = w({}, {'logps': torch.tensor([1.0, 2.0, 3.0], requires_grad=True)})
        assert result['loss'].item() == 6.0

    def test_exception_returns_zero_loss(self):
        w = safe_loss(ExplodingLoss())
        result = w({}, {'logps': torch.tensor([1.0], requires_grad=True)})
        assert result['loss'].item() == 0.0
        assert result['num_tokens'] == 0

    def test_zero_loss_is_graph_connected(self):
        w = safe_loss(ExplodingLoss())
        logps = torch.tensor([1.0, 2.0], requires_grad=True)
        result = w({}, {'logps': logps})
        result['loss'].backward()
        assert logps.grad is not None

    def test_backward_on_zero_loss_yields_zero_grad(self):
        w = safe_loss(ExplodingLoss())
        logps = torch.randn(3, requires_grad=True)
        result = w({}, {'logps': logps})
        result['loss'].backward()
        assert (logps.grad == 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Unit Tests: _zero_loss
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLoss:

    def test_from_logps(self):
        logps = torch.tensor([1.0, 2.0], requires_grad=True)
        r = _zero_loss({'logps': logps})
        assert r['loss'].item() == 0.0
        r['loss'].backward()
        assert logps.grad is not None

    def test_from_logits(self):
        logits = torch.randn(2, 3, requires_grad=True)
        r = _zero_loss({'logits': logits})
        assert r['loss'].item() == 0.0
        r['loss'].backward()
        assert logits.grad is not None

    def test_from_loss_key(self):
        t = torch.tensor(1.0, requires_grad=True)
        r = _zero_loss({'loss': t})
        assert r['loss'].item() == 0.0
        r['loss'].backward()
        assert t.grad is not None

    def test_fallback_no_grad_tensor(self):
        r = _zero_loss({'logps': torch.tensor([1.0])})  # no requires_grad
        assert r['loss'].item() == 0.0
        assert r['loss'].requires_grad

    def test_empty_dict(self):
        r = _zero_loss({})
        assert r['loss'].item() == 0.0

    def test_non_dict(self):
        r = _zero_loss('not_a_dict')
        assert r['loss'].item() == 0.0

    def test_num_tokens_zero(self):
        r = _zero_loss({'logps': torch.tensor([1.0], requires_grad=True)})
        assert r['num_tokens'] == 0

    def test_priority_order_logps_first(self):
        """logps should be preferred over logits for graph connectivity."""
        logps = torch.tensor([1.0], requires_grad=True)
        logits = torch.randn(1, 3, requires_grad=True)
        r = _zero_loss({'logps': logps, 'logits': logits})
        r['loss'].backward()
        assert logps.grad is not None
        assert logits.grad is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. Unit Tests: @nccl_safe decorator
# ═══════════════════════════════════════════════════════════════════════════


def _make_model(adapter_name='default', outputs=None, loss_value='sentinel'):
    """Create a mock model with optimizer_group for decorator tests."""
    model = MagicMock()
    ts = TrainStatus()
    ts.outputs = outputs
    ts.loss_value = loss_value

    og = MagicMock()
    og.train_status = ts
    model.optimizer_group = {adapter_name: og}
    model._get_default_group = MagicMock(return_value=adapter_name)
    return model, og


class TestNcclSafeDecorator:

    # ── Basic behaviour ──

    def test_transparent_in_dev_mode(self, _dev_mode):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            return {'loss': 1.0}

        model, _ = _make_model()
        assert method(model, inputs=[], adapter_name='default') == {'loss': 1.0}

    def test_normal_call_passes(self):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            return {'loss': 1.0}

        model, _ = _make_model()
        assert method(model, inputs=[], adapter_name='default') == {'loss': 1.0}

    # ── Pre-forward failure -> re-raise ──

    def test_pre_forward_error_propagates(self):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            raise ValueError('pre-forward')

        model, _ = _make_model(outputs=None)
        with pytest.raises(ValueError, match='pre-forward'):
            method(model, inputs=[], adapter_name='default')

    # ── Post-forward, pre-backward -> force backward ──

    def test_post_forward_pre_backward_forces_backward(self):
        outputs_after = {'logps': torch.tensor([1.0], requires_grad=True)}

        @nccl_safe
        def method(self, *, inputs, **kwargs):
            og = self.optimizer_group['default']
            og.train_status.outputs = outputs_after
            og.train_status.loss_value = torch.tensor(1.0)
            raise RuntimeError('mid-pipeline')

        model, _ = _make_model(outputs=None, loss_value=None)
        model.backward = MagicMock()
        model.model = MagicMock()
        model.model.parameters = MagicMock(
            return_value=iter([torch.randn(3, requires_grad=True)]))

        result = method(model, inputs=[], adapter_name='default')
        model.backward.assert_called_once()
        assert result['loss'] == 0.0

    # ── Post-backward failure -> no extra backward ──

    def test_post_backward_no_extra_backward(self):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            og = self.optimizer_group['default']
            og.train_status.outputs = {'logps': torch.tensor([1.0])}
            og.train_status.loss_value = None  # backward done
            raise RuntimeError('post-backward')

        model, _ = _make_model(outputs=None)
        model.backward = MagicMock()

        result = method(model, inputs=[], adapter_name='default')
        model.backward.assert_not_called()
        assert result['loss'] == 0.0

    # ── tinker mode ──

    def test_tinker_returns_list(self):
        @nccl_safe(tinker=True)
        def method(self, *, inputs, **kwargs):
            og = self.optimizer_group['default']
            og.train_status.outputs = {'logps': torch.tensor([1.0])}
            og.train_status.loss_value = None
            raise RuntimeError('err')

        model, _ = _make_model(outputs=None)
        assert method(model, inputs=[], adapter_name='default') == [[], 0.0]

    # ── No optimizer group -> passthrough ──

    def test_no_optimizer_group_raises(self):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            raise ValueError('err')

        model = MagicMock()
        model.optimizer_group = {}
        model._get_default_group = MagicMock(return_value='default')

        with pytest.raises(ValueError, match='err'):
            method(model, inputs=[], adapter_name='default')

    # ── gradient_accumulation_steps forwarded ──

    def test_gas_forwarded_to_backward(self):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            og = self.optimizer_group['default']
            og.train_status.outputs = {'logps': torch.tensor([1.0], requires_grad=True)}
            og.train_status.loss_value = torch.tensor(1.0)
            raise RuntimeError('err')

        model, _ = _make_model(outputs=None, loss_value=None)
        model.backward = MagicMock()
        model.model = MagicMock()
        model.model.parameters = MagicMock(
            return_value=iter([torch.randn(3, requires_grad=True)]))

        method(model, inputs=[], adapter_name='default', gradient_accumulation_steps=4)
        _, call_kwargs = model.backward.call_args
        assert call_kwargs.get('gradient_accumulation_steps') == 4


# ═══════════════════════════════════════════════════════════════════════════
# 5. Unit Tests: BaseOptimizerGroup.__setattr__
# ═══════════════════════════════════════════════════════════════════════════


class TestOptimizerGroupSetattr:

    def test_auto_wraps_loss(self):
        og = BaseOptimizerGroup()
        loss = DummyLoss()
        og.loss_instance = loss
        assert og.loss_instance is not loss
        assert og.loss_instance._nccl_safe_wrapped is True

    def test_does_not_wrap_none(self):
        og = BaseOptimizerGroup()
        og.loss_instance = None
        assert og.loss_instance is None

    def test_other_attrs_unaffected(self):
        og = BaseOptimizerGroup()
        og.adapter_name = 'test'
        assert og.adapter_name == 'test'
        og.cur_step = 42
        assert og.cur_step == 42

    def test_idempotent_via_setattr(self):
        og = BaseOptimizerGroup()
        og.loss_instance = DummyLoss()
        first = og.loss_instance
        og.loss_instance = first
        assert og.loss_instance is first

    def test_transparent_in_dev_mode(self, _dev_mode):
        """In dev mode, OG auto-wraps but wrapper propagates exceptions."""
        og = BaseOptimizerGroup()
        loss = ExplodingLoss()
        og.loss_instance = loss
        assert og.loss_instance is not loss  # always wrapped now
        assert og.loss_instance._nccl_safe_wrapped is True
        with pytest.raises(RuntimeError, match='Simulated'):
            og.loss_instance({}, {'logps': torch.tensor([1.0])})


# ═══════════════════════════════════════════════════════════════════════════
# 6. Integration: real loss functions through safe_loss
# ═══════════════════════════════════════════════════════════════════════════


def _grpo_fixtures(batch=2, seq_len=8):
    """Create valid GRPO inputs/outputs/kwargs."""
    labels = torch.randint(0, 100, (batch, seq_len))
    labels[:, :3] = -100
    inputs = {'labels': labels}
    logps = torch.randn(batch, seq_len, requires_grad=True)
    outputs = {'logps': logps}
    n_valid = (labels != -100).sum(dim=1).tolist()
    old_logps = [torch.randn(n).tolist() for n in n_valid]
    advantages = torch.randn(batch).tolist()
    return inputs, outputs, old_logps, advantages


class TestIntegrationRealLosses:

    # ── GRPO ──

    def test_grpo_normal(self):
        w = safe_loss(GRPOLoss(epsilon=0.2))
        inp, out, olp, adv = _grpo_fixtures()
        r = w(inp, out, old_logps=olp, advantages=adv)
        assert r['loss'].requires_grad

    def test_grpo_bad_old_logps_caught(self):
        """old_logps length mismatch -> AssertionError -> caught."""
        w = safe_loss(GRPOLoss(epsilon=0.2))
        inp, out, _, adv = _grpo_fixtures()
        r = w(inp, out, old_logps=[[0.1, 0.2]], advantages=adv)
        assert r['loss'].item() == 0.0

    def test_grpo_bad_old_logps_graph_connected(self):
        """Zero loss from GRPO error should still be graph-connected."""
        w = safe_loss(GRPOLoss(epsilon=0.2))
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        labels[0, :2] = -100
        logps = torch.randn(1, 5, requires_grad=True)
        r = w({'labels': labels}, {'logps': logps},
              old_logps=[[0.1, 0.2]], advantages=[1.0])
        r['loss'].backward()
        assert logps.grad is not None

    # ── CrossEntropy ──

    def test_cross_entropy_normal(self):
        w = safe_loss(CrossEntropyLoss())
        labels = torch.randint(0, 100, (2, 8))
        labels[:, :3] = -100
        logps = torch.randn(2, 8, requires_grad=True)
        r = w({'labels': labels}, {'logps': logps})
        assert r['loss'].requires_grad

    def test_cross_entropy_missing_labels_caught(self):
        w = safe_loss(CrossEntropyLoss())
        r = w({}, {'logps': torch.randn(2, 8, requires_grad=True)})
        assert r['loss'].item() == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. Integration: full chain via OptimizerGroup
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndOptimizerGroup:

    def test_grpo_via_og(self):
        og = BaseOptimizerGroup()
        og.loss_instance = GRPOLoss(epsilon=0.2)
        inp, out, olp, adv = _grpo_fixtures()
        r = og.loss_instance(inp, out, old_logps=olp, advantages=adv)
        assert 'loss' in r and r['loss'].requires_grad

    def test_grpo_bad_data_via_og(self):
        og = BaseOptimizerGroup()
        og.loss_instance = GRPOLoss(epsilon=0.2)
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        labels[0, :2] = -100  # 3 valid positions
        logps = torch.randn(1, 5, requires_grad=True)
        # 2 values for 3 valid positions -> AssertionError in _pad_and_align_to_batch
        r = og.loss_instance({'labels': labels}, {'logps': logps},
                             old_logps=[[0.1, 0.2]], advantages=[1.0])
        assert r['loss'].item() == 0.0

    def test_replace_loss_auto_wraps(self):
        og = BaseOptimizerGroup()
        og.loss_instance = DummyLoss()
        first = og.loss_instance

        og.loss_instance = ExplodingLoss()
        second = og.loss_instance

        assert first is not second
        assert second._nccl_safe_wrapped is True
        r = second({}, {'logps': torch.tensor([1.0], requires_grad=True)})
        assert r['loss'].item() == 0.0

    def test_ce_via_og(self):
        og = BaseOptimizerGroup()
        og.loss_instance = CrossEntropyLoss()
        labels = torch.randint(0, 100, (2, 8))
        labels[:, :3] = -100
        r = og.loss_instance({'labels': labels},
                             {'logps': torch.randn(2, 8, requires_grad=True)})
        assert 'loss' in r


# ═══════════════════════════════════════════════════════════════════════════
# 8. Adversarial: monkey-patch injection
# ═══════════════════════════════════════════════════════════════════════════


class TestAdversarial:

    def test_loss_raises_runtime_error(self):
        """Loss that raises RuntimeError is caught."""
        class OOMLoss(Loss):
            def __call__(self, inputs, outputs, **kwargs):
                raise RuntimeError('GPU OOM')

        w = safe_loss(OOMLoss())
        r = w({}, {'logps': torch.tensor([1.0], requires_grad=True)})
        assert r['loss'].item() == 0.0

    def test_original_grpo_assertion_error(self):
        """The original bug: AssertionError in _pad_and_align_to_batch."""
        w = safe_loss(GRPOLoss(epsilon=0.2))
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        labels[0, :2] = -100
        logps = torch.randn(1, 5, requires_grad=True)
        # 2 values but 3 valid positions -> AssertionError
        r = w({'labels': labels}, {'logps': logps},
              old_logps=[[0.1, 0.2]], advantages=[1.0])
        assert r['loss'].item() == 0.0
        r['loss'].backward()
        assert logps.grad is not None

    def test_nan_passes_through(self):
        """NaN loss is NOT an exception -- should NOT be caught."""
        class NanLoss(Loss):
            def __call__(self, inputs, outputs, **kw):
                return LossOutput(
                    loss=torch.tensor(float('nan'), requires_grad=True),
                    num_tokens=1)

        w = safe_loss(NanLoss())
        r = w({}, {'logps': torch.tensor([1.0], requires_grad=True)})
        assert torch.isnan(r['loss'])

    def test_consecutive_errors_all_caught(self):
        w = safe_loss(ExplodingLoss())
        for _ in range(10):
            r = w({}, {'logps': torch.randn(3, requires_grad=True)})
            assert r['loss'].item() == 0.0

    def test_error_then_normal(self):
        call_count = [0]

        class FlakeyLoss(Loss):
            def __call__(self, inputs, outputs, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError('first call fails')
                return LossOutput(loss=outputs['logps'].sum(), num_tokens=1)

        w = safe_loss(FlakeyLoss())
        out1 = {'logps': torch.tensor([1.0, 2.0], requires_grad=True)}
        r1 = w({}, out1)
        assert r1['loss'].item() == 0.0

        out2 = {'logps': torch.tensor([4.0, 5.0], requires_grad=True)}
        r2 = w({}, out2)
        assert r2['loss'].item() == 9.0

    def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt is BaseException, NOT caught."""
        class KBLoss(Loss):
            def __call__(self, *a, **kw):
                raise KeyboardInterrupt()

        w = safe_loss(KBLoss())
        with pytest.raises(KeyboardInterrupt):
            w({}, {'logps': torch.tensor([1.0], requires_grad=True)})

    def test_system_exit_propagates(self):
        class ExitLoss(Loss):
            def __call__(self, *a, **kw):
                raise SystemExit(1)

        w = safe_loss(ExitLoss())
        with pytest.raises(SystemExit):
            w({}, {'logps': torch.tensor([1.0], requires_grad=True)})


# ═══════════════════════════════════════════════════════════════════════════
# 9. Backward Compatibility (dev mode transparency)
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompat:

    def test_safe_loss_transparent(self, _dev_mode):
        loss = ExplodingLoss()
        wrapped = safe_loss(loss)
        assert wrapped is not loss  # always wrapped, but transparent in dev mode
        with pytest.raises(RuntimeError, match='Simulated'):
            wrapped({}, {'logps': torch.tensor([1.0])})

    def test_nccl_safe_transparent(self, _dev_mode):
        @nccl_safe
        def method(self, *, inputs, **kwargs):
            raise ValueError('should propagate')

        with pytest.raises(ValueError, match='should propagate'):
            method(MagicMock(), inputs=[])

    def test_og_transparent(self, _dev_mode):
        og = BaseOptimizerGroup()
        loss = ExplodingLoss()
        og.loss_instance = loss
        assert og.loss_instance is not loss  # always wrapped
        assert og.loss_instance._nccl_safe_wrapped is True
        with pytest.raises(RuntimeError, match='Simulated'):
            og.loss_instance({}, {'logps': torch.tensor([1.0])})
