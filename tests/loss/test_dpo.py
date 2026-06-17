# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for DPO family losses: DPO, SimPO, CPO, ORPO."""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import DPOLoss, SimPOLoss, CPOLoss, ORPOLoss


def _make_preference_batch(batch_size=4, seq_len=8, vocab_size=20, with_ref=True):
    """Create a synthetic DPO-style interleaved batch.

    Layout: [chosen_0, rejected_0, chosen_1, rejected_1, ...]
    """
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mark some positions as ignore
    for i in range(batch_size):
        labels[i, seq_len // 2:] = -100

    logps = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    inputs = {'labels': labels}
    outputs = {'logps': logps, 'logits': logits}

    ref_logps = None
    if with_ref:
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        ref_logps = F.log_softmax(ref_logits, dim=-1).gather(
            -1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    return inputs, outputs, ref_logps


class TestDPOLoss:

    def test_basic_dpo_sigmoid(self):
        loss_fn = DPOLoss(beta=0.1, loss_type='sigmoid')
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0

    def test_dpo_hinge(self):
        loss_fn = DPOLoss(beta=0.1, loss_type='hinge')
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert result['loss'].dim() == 0

    def test_dpo_ipo(self):
        loss_fn = DPOLoss(beta=0.1, loss_type='ipo')
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert result['loss'].dim() == 0

    def test_dpo_kto_pair(self):
        loss_fn = DPOLoss(beta=0.1, loss_type='kto_pair')
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert result['loss'].dim() == 0

    def test_dpo_reference_free(self):
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert result['loss'].dim() == 0

    def test_dpo_no_ref_returns_zero(self):
        """Without ref_logps and not reference_free, loss should be zero."""
        loss_fn = DPOLoss(beta=0.1)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_dpo_with_label_smoothing(self):
        loss_fn = DPOLoss(beta=0.1, label_smoothing=0.1, loss_type='sigmoid')
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert result['loss'].dim() == 0

    def test_dpo_with_sft_weight(self):
        loss_fn = DPOLoss(beta=0.1, sft_weight=0.1)
        inputs, outputs, ref_logps = _make_preference_batch()
        result = loss_fn(inputs, outputs, ref_logps=ref_logps)
        assert result['loss'].dim() == 0

    def test_dpo_precomputed_ref_chosen_rejected(self):
        loss_fn = DPOLoss(beta=0.1)
        inputs, outputs, ref_logps = _make_preference_batch()
        # Compute sequence-level ref logps for chosen/rejected
        labels = inputs['labels']
        loss_mask = (labels != -100).float()
        chosen_ref = (ref_logps[0::2] * loss_mask[0::2]).sum(dim=-1)
        rejected_ref = (ref_logps[1::2] * loss_mask[1::2]).sum(dim=-1)
        result = loss_fn(inputs, outputs, ref_chosen_logps=chosen_ref, ref_rejected_logps=rejected_ref)
        assert result['loss'].dim() == 0

    def test_dpo_invalid_loss_type(self):
        with pytest.raises(ValueError, match='Unknown loss_type'):
            DPOLoss(loss_type='invalid_type')

    def test_dpo_label_smoothing_non_sigmoid_raises(self):
        with pytest.raises(ValueError, match='label_smoothing'):
            DPOLoss(label_smoothing=0.1, loss_type='hinge')

    def test_dpo_odd_batch_raises(self):
        loss_fn = DPOLoss(beta=0.1)
        labels = torch.randint(0, 10, (3, 5))
        logps = torch.randn(3, 5)
        with pytest.raises(AssertionError, match='even'):
            loss_fn({'labels': labels}, {'logps': logps})

    def test_dpo_from_logits(self):
        """Test with logits instead of pre-computed logps."""
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        # Remove logps, force computation from logits
        outputs_no_logps = {'logits': outputs['logits']}
        result = loss_fn(inputs, outputs_no_logps)
        assert result['loss'].dim() == 0


class TestSimPOLoss:

    def test_basic_simpo(self):
        loss_fn = SimPOLoss(beta=2.5, gamma=0.5)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0

    def test_simpo_loss_non_negative(self):
        loss_fn = SimPOLoss()
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        # -log(sigmoid(x)) is always >= 0
        assert result['loss'].item() >= 0

    def test_simpo_odd_batch_raises(self):
        loss_fn = SimPOLoss()
        labels = torch.randint(0, 10, (3, 5))
        logps = torch.randn(3, 5)
        with pytest.raises(AssertionError):
            loss_fn({'labels': labels}, {'logps': logps})


class TestCPOLoss:

    def test_basic_cpo(self):
        loss_fn = CPOLoss(beta=0.1, bc_coef=1.0)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0

    def test_cpo_zero_bc_coef(self):
        loss_fn = CPOLoss(beta=0.1, bc_coef=0.0)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert result['loss'].dim() == 0


class TestORPOLoss:

    def test_basic_orpo(self):
        loss_fn = ORPOLoss(lambda_orpo=0.1)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0

    def test_orpo_loss_finite(self):
        loss_fn = ORPOLoss(lambda_orpo=0.1)
        inputs, outputs, _ = _make_preference_batch(with_ref=False)
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result['loss'])

    def test_orpo_odd_batch_raises(self):
        loss_fn = ORPOLoss()
        labels = torch.randint(0, 10, (3, 5))
        logps = torch.randn(3, 5)
        with pytest.raises(AssertionError):
            loss_fn({'labels': labels}, {'logps': logps})
