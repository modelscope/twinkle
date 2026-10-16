# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the classification-head losses: ``SeqClsLoss`` and ``RewardLoss``.

Both consume a per-SAMPLE score head (``outputs['logits']`` already reduced to one row per
sequence) rather than per-token logps, so they are grouped here:

* ``SeqClsLoss`` scores each sequence against a fixed label set, dispatching by a REQUIRED
  ``problem_type`` exactly as HF ``*ForSequenceClassification`` does:
  regression -> MSE, single_label_classification -> CE, multi_label_classification -> BCE.
* ``RewardLoss`` is the pairwise Bradley-Terry RM objective over interleaved
  ``[chosen, rejected]`` scores, with an optional centring regulariser.

Every case drives the real ``__call__`` with ``backward()`` and numeric assertions against
closed-form / manual references. CPU only, no stubs.
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from twinkle.loss import RewardLoss, SeqClsLoss, torch_loss_mapping


def test_losses_registered_in_mapping():
    assert torch_loss_mapping['seq_cls'] is SeqClsLoss
    assert torch_loss_mapping['reward'] is RewardLoss
    assert torch_loss_mapping['rm'] is RewardLoss


# ── SeqClsLoss ────────────────────────────────────────────────────────────────


class TestSeqClsLoss:

    def test_requires_logits_not_logps(self):
        assert SeqClsLoss.require_logits is True
        assert SeqClsLoss.require_logps is False

    def test_problem_type_is_required_and_validated(self):
        with pytest.raises(ValueError, match='problem_type must be one of'):
            SeqClsLoss(problem_type='unknown_type', num_labels=2)

    def test_regression_num_labels_one_known_value(self):
        """num_labels==1 squeezes the trailing axis; MSE of zeros vs ones is exactly 1.0."""
        logits = torch.zeros(3, 1)
        labels = torch.ones(3)
        result = SeqClsLoss('regression', num_labels=1)({'labels': labels}, {'logits': logits})
        assert result['loss'].item() == pytest.approx(1.0, abs=1e-6)
        assert result['num_tokens'] == 0

    def test_regression_multi_output_matches_manual_mse(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        labels = torch.randn(4, 3)
        result = SeqClsLoss('regression', num_labels=3)({'labels': labels}, {'logits': logits})
        expected = nn.MSELoss()(logits, labels)
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_single_label_matches_manual_cross_entropy(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 4)
        labels = torch.randint(0, 4, (5,))
        result = SeqClsLoss('single_label_classification', num_labels=4)(
            {'labels': labels}, {'logits': logits})
        expected = nn.CrossEntropyLoss()(logits, labels)
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_single_label_confident_correct_near_zero(self):
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        labels = torch.tensor([0, 1])
        result = SeqClsLoss('single_label_classification', num_labels=2)(
            {'labels': labels}, {'logits': logits})
        assert result['loss'].item() < 1e-3

    def test_multi_label_matches_manual_bce(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 3)
        labels = torch.randint(0, 2, (4, 3)).float()
        result = SeqClsLoss('multi_label_classification', num_labels=3)(
            {'labels': labels}, {'logits': logits})
        expected = nn.BCEWithLogitsLoss()(logits, labels)
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    @pytest.mark.parametrize('problem_type,logits_shape,label_kind', [
        ('regression', (4, 1), 'float'),
        ('single_label_classification', (4, 3), 'int'),
        ('multi_label_classification', (4, 3), 'multihot'),
    ])
    def test_gradient_flows_for_every_problem_type(self, problem_type, logits_shape, label_kind):
        torch.manual_seed(0)
        logits = torch.randn(*logits_shape, requires_grad=True)
        if label_kind == 'float':
            labels = torch.randn(4)
        elif label_kind == 'int':
            labels = torch.randint(0, logits_shape[1], (4,))
        else:
            labels = torch.randint(0, 2, (4, logits_shape[1])).float()
        result = SeqClsLoss(problem_type, num_labels=logits_shape[1])(
            {'labels': labels}, {'logits': logits})
        result['loss'].backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0


# ── RewardLoss (pairwise Bradley-Terry) ───────────────────────────────────────


class TestRewardLoss:

    def test_requires_logits_not_logps(self):
        assert RewardLoss.require_logits is True
        assert RewardLoss.require_logps is False

    def test_tied_scores_known_value(self):
        """chosen == rejected -> -log(sigmoid(0)) = ln 2."""
        logits = torch.zeros(2, 1)
        result = RewardLoss()({'labels': torch.zeros(2)}, {'logits': logits})
        assert result['loss'].item() == pytest.approx(math.log(2.0), abs=1e-6)
        assert result['num_tokens'] == 0

    def test_matches_manual_bradley_terry(self):
        torch.manual_seed(0)
        scores = torch.randn(6)  # interleaved chosen/rejected
        logits = scores.unsqueeze(-1)  # head emits [B, 1]
        result = RewardLoss()({'labels': torch.zeros(6)}, {'logits': logits})
        chosen, rejected = scores[0::2], scores[1::2]
        expected = -F.logsigmoid(chosen - rejected).mean()
        assert torch.allclose(result['loss'], expected, atol=1e-6)

    def test_chosen_dominates_rejected_near_zero_loss(self):
        logits = torch.tensor([[20.0], [-20.0]])
        result = RewardLoss()({'labels': torch.zeros(2)}, {'logits': logits})
        assert result['loss'].item() < 1e-6

    def test_odd_batch_raises_assertion(self):
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        with pytest.raises(AssertionError, match='even batch'):
            RewardLoss()({'labels': torch.zeros(3)}, {'logits': logits})

    def test_center_rewards_coefficient_adds_regulariser(self):
        """coef>0 adds coef*mean((chosen+rejected)^2); with tied nonzero scores it is separable."""
        logits = torch.tensor([[1.0], [1.0]])  # chosen=1, rejected=1
        plain = RewardLoss()({'labels': torch.zeros(2)}, {'logits': logits})['loss']
        centred = RewardLoss(center_rewards_coefficient=0.5)(
            {'labels': torch.zeros(2)}, {'logits': logits})['loss']
        # BT part is ln 2 for both; centring adds 0.5 * mean((1+1)^2) = 0.5 * 4 = 2.0
        assert plain.item() == pytest.approx(math.log(2.0), abs=1e-6)
        assert centred.item() == pytest.approx(math.log(2.0) + 2.0, abs=1e-6)

    def test_gradient_flows_to_scores(self):
        logits = torch.randn(6, 1, requires_grad=True)
        result = RewardLoss()({'labels': torch.zeros(6)}, {'logits': logits})
        result['loss'].backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
