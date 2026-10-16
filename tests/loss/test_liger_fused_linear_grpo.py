# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for ``LigerFusedLinearGRPOLoss`` (``liger_fused_linear_grpo.py``).

This loss subclasses ``GRPOLoss`` and fuses the final ``lm_head`` matmul with the GRPO
objective so the ``(B, T, V)`` logits are never materialised. Two paths must both be correct:

* **Fused path** -- Liger's chunked kernel computes per-token log-probs from the hidden states
  and ``lm_head.weight`` internally. Verified against an independent ``GRPOLoss`` on logits we
  materialise ourselves with ``F.linear`` (the correctness assertion is ours, not the library's).
  Liger's fused GRPO kernel runs on CPU as well as CUDA, so this needs no accelerator.
* **Defensive fallback** -- when the fused path is unusable the loss degrades transparently to the
  pure-torch GRPO objective, materialising logits via ``F.linear(hidden, lm_head.weight)``. It is
  skipped when ``outputs['lm_head']`` is absent, when ``advantages is None``, or when
  ``liger_kernel`` is missing / the kernel raises (simulated here by monkeypatching the lazy
  import to raise ``ImportError``).

No stubs of the loss internals; every case drives the real ``__call__`` with ``backward()``.
"""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import twinkle.loss.liger_fused_linear_grpo as lflg
from twinkle.loss import GRPOLoss, LigerFusedLinearGRPOLoss, torch_loss_mapping


def _make_grpo_batch(bs=2, seq=6, hidden=16, vocab=32, seed=0):
    """A synthetic fused-GRPO step: hidden states (logits under the patch), labels, advantages."""
    torch.manual_seed(seed)
    hidden_states = torch.randn(bs, seq, hidden)
    labels = torch.randint(0, vocab, (bs, seq))
    labels[:, seq // 2:] = -100  # non-response tokens ignored
    advantages = torch.randn(bs, 1)
    return hidden_states, labels, advantages


def _lm_head(hidden, vocab, seed=1):
    torch.manual_seed(seed)
    return nn.Linear(hidden, vocab, bias=False)


def _base_grpo_reference(hidden, head, labels, **kwargs):
    """Materialise logits in torch and run the base GRPO objective -- the independent reference."""
    manual_logits = F.linear(hidden, head.weight)
    base = GRPOLoss(epsilon=kwargs.get('epsilon', 0.2), beta=kwargs.get('beta', 0.0))
    return base({'labels': labels}, {'logits': manual_logits}, **kwargs)


# ── registration + flags ──────────────────────────────────────────────────────


def test_loss_registered_in_mapping():
    assert torch_loss_mapping['liger_fused_linear_grpo'] is LigerFusedLinearGRPOLoss


def test_loss_flags_keep_logits_skip_logps():
    assert LigerFusedLinearGRPOLoss.require_logits is True
    assert LigerFusedLinearGRPOLoss.require_logps is False


def test_subclasses_grpo_loss():
    assert issubclass(LigerFusedLinearGRPOLoss, GRPOLoss)


# ── Fused path: the Liger kernel runs and matches the torch objective ─────────


class TestFusedPath:

    def test_fused_runs_and_matches_materialised_grpo(self):
        """lm_head present + advantages -> fused kernel; result equals base GRPO on F.linear logits."""
        hidden, labels, advantages = _make_grpo_batch()
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        out = fused({'labels': labels}, {'logits': hidden.clone(), 'lm_head': head}, advantages=advantages)

        assert fused._fused_broken is False, 'the fused kernel should run on CPU, no fallback expected'
        assert torch.isfinite(out['loss'])
        expected = _base_grpo_reference(hidden, head, labels, advantages=advantages)
        assert torch.allclose(out['loss'], expected['loss'], atol=1e-5)

    def test_fused_backprops_into_hidden_and_head(self):
        hidden, labels, advantages = _make_grpo_batch()
        hidden = hidden.clone().requires_grad_(True)
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        out = fused({'labels': labels}, {'logits': hidden, 'lm_head': head}, advantages=advantages)
        out['loss'].backward()
        assert fused._fused_broken is False
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()

    def test_fused_with_old_and_ref_logps_kl_penalty(self):
        """beta>0 with ref_logps exercises the fused KL path (use_ref_model=True)."""
        torch.manual_seed(0)
        hidden, labels, advantages = _make_grpo_batch(bs=2, seq=6, hidden=16, vocab=32)
        head = _lm_head(hidden.shape[-1], 32)
        old_logps = torch.randn(2, 6)
        ref_logps = torch.randn(2, 6)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.04)
        out = fused(
            {'labels': labels},
            {'logits': hidden.clone(), 'lm_head': head},
            old_logps=old_logps,
            ref_logps=ref_logps,
            advantages=advantages)
        assert fused._fused_broken is False
        assert torch.isfinite(out['loss'])


# ── No lm_head: outputs['logits'] are already real -> base GRPO directly ──────


class TestNoLmHeadFallback:

    def test_matches_base_grpo_on_real_logits(self):
        torch.manual_seed(0)
        bs, seq, vocab = 2, 6, 32
        logits = torch.randn(bs, seq, vocab)
        labels = torch.randint(0, vocab, (bs, seq))
        labels[:, seq // 2:] = -100
        advantages = torch.randn(bs, 1)

        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        base = GRPOLoss(epsilon=0.2, beta=0.0)
        out_fused = fused({'labels': labels}, {'logits': logits.clone()}, advantages=advantages)
        out_base = base({'labels': labels}, {'logits': logits.clone()}, advantages=advantages)
        assert torch.allclose(out_fused['loss'], out_base['loss'], atol=1e-6)
        assert fused._fused_broken is False  # the fused path was never attempted

    def test_no_advantages_returns_zero(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 6, 32)
        labels = torch.randint(0, 32, (2, 6))
        fused = LigerFusedLinearGRPOLoss()
        out = fused({'labels': labels}, {'logits': logits}, advantages=None)
        assert out['loss'].item() == pytest.approx(0.0, abs=1e-8)


# ── Defensive fallback: liger unavailable -> materialise logits + base GRPO ───


class TestDefensiveFallback:

    @pytest.fixture
    def liger_missing(self, monkeypatch):
        """Simulate liger_kernel being absent: the lazy import raises ImportError."""

        def _boom():
            raise ImportError('simulated: liger_kernel not installed')

        monkeypatch.setattr(lflg, '_get_liger_module', _boom)
        # Reset the module-level cache so the patched function is actually consulted.
        monkeypatch.setattr(lflg, '_LigerFLGRPOModule', None)

    def test_missing_liger_materialises_and_matches_grpo(self, liger_missing):
        hidden, labels, advantages = _make_grpo_batch()
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        out = fused({'labels': labels}, {'logits': hidden.clone(), 'lm_head': head}, advantages=advantages)

        assert fused._fused_broken is True, 'a raising kernel must trip the defensive fallback'
        assert fused._warned is True
        assert torch.isfinite(out['loss'])
        expected = _base_grpo_reference(hidden, head, labels, advantages=advantages)
        assert torch.allclose(out['loss'], expected['loss'], atol=1e-5)

    def test_materialised_fallback_backprops_into_hidden_and_head(self, liger_missing):
        hidden, labels, advantages = _make_grpo_batch()
        hidden = hidden.clone().requires_grad_(True)
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        out = fused({'labels': labels}, {'logits': hidden, 'lm_head': head}, advantages=advantages)
        out['loss'].backward()
        assert fused._fused_broken is True
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()

    def test_second_call_skips_fused_after_break(self, liger_missing):
        """Once broken, later calls go straight to the fallback without re-attempting the kernel."""
        hidden, labels, advantages = _make_grpo_batch()
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss(epsilon=0.2, beta=0.0)
        first = fused({'labels': labels}, {'logits': hidden.clone(), 'lm_head': head}, advantages=advantages)
        assert fused._fused_broken is True
        second = fused({'labels': labels}, {'logits': hidden.clone(), 'lm_head': head}, advantages=advantages)
        assert torch.allclose(first['loss'], second['loss'], atol=1e-6)

    def test_lm_head_present_but_no_advantages_skips_fused_and_returns_zero(self):
        """advantages is None -> fused skipped (not broken), logits materialised, base GRPO -> zero."""
        hidden, labels, _ = _make_grpo_batch()
        head = _lm_head(hidden.shape[-1], 32)
        fused = LigerFusedLinearGRPOLoss()
        out = fused({'labels': labels}, {'logits': hidden, 'lm_head': head}, advantages=None)
        assert out['loss'].item() == pytest.approx(0.0, abs=1e-8)
        assert fused._fused_broken is False  # fused was never attempted (advantages is None)
