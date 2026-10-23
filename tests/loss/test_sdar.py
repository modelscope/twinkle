# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for SDAR -- the confidence-gated teacher-distillation auxiliary objective."""
import pytest
import torch

from twinkle.loss import GRPOLoss
from twinkle.loss.sdar import compute_sdar_loss


def _make_sdar_batch():
    student = torch.tensor([[-0.5, -1.0, -0.8], [-0.3, -0.6, 0.0]])
    teacher = torch.tensor([[-0.2, -1.2, -0.4], [-0.5, -0.4, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    return student, teacher, mask


def _manual(student, teacher, mask, gate_beta=5.0):
    """Independent oracle for the gated k3 term and its metrics."""
    delta = teacher - student.detach()
    gate = torch.sigmoid(gate_beta * delta)
    gated_kl = gate * (teacher - student)
    mask_sum = mask.sum().clamp(min=1)
    loss = (gated_kl * mask).sum() / mask_sum
    metrics = {
        'sdar/gate_mean': ((gate * mask).sum() / mask_sum).item(),
        'sdar/gate_active_ratio': ((gate > 0.5).float() * mask).sum().item() / mask_sum.item(),
        'sdar/teacher_gap_mean': ((delta * mask).sum() / mask_sum).item(),
        'sdar/loss': loss.item(),
    }
    return loss, gated_kl, gate, metrics


class TestComputeSDARLoss:

    def test_zero_when_teacher_equals_student(self):
        student = torch.full((2, 3), -0.7)
        teacher = student.clone()
        mask = torch.ones(2, 3, dtype=torch.bool)

        loss, _ = compute_sdar_loss(student, teacher, mask)

        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_matches_manual_gated_k3(self):
        student, teacher, mask = _make_sdar_batch()

        loss, _ = compute_sdar_loss(student, teacher, mask, gate_beta=5.0)

        expected, _, _, _ = _manual(student, teacher, mask)
        torch.testing.assert_close(loss, expected)

    def test_gate_and_teacher_are_detached(self):
        student, teacher, mask = _make_sdar_batch()
        student.requires_grad_(True)
        teacher.requires_grad_(True)

        loss, _ = compute_sdar_loss(student, teacher, mask)
        loss.backward()

        assert student.grad is not None
        assert teacher.grad is None

    def test_metrics_match_manual(self):
        student, teacher, mask = _make_sdar_batch()

        _, metrics = compute_sdar_loss(student, teacher, mask, gate_beta=5.0)

        _, _, _, expected = _manual(student, teacher, mask)
        assert set(metrics) == set(expected)
        for key, value in expected.items():
            assert metrics[key] == pytest.approx(value, rel=1e-6), key

    def test_masked_tokens_do_not_contribute(self):
        student, teacher, mask = _make_sdar_batch()
        polluted = teacher.clone()
        polluted[~mask] += 123.0

        baseline, baseline_metrics = compute_sdar_loss(student, teacher, mask)
        perturbed, perturbed_metrics = compute_sdar_loss(student, polluted, mask)

        torch.testing.assert_close(perturbed, baseline)
        for key, value in baseline_metrics.items():
            assert perturbed_metrics[key] == pytest.approx(value), key

    def test_aggregation_modes(self):
        student, teacher, mask = _make_sdar_batch()
        _, gated_kl, _, _ = _manual(student, teacher, mask)
        mask_sum = mask.sum().clamp(min=1)
        expected = {
            'token-mean': (gated_kl * mask).sum() / mask_sum,
            'seq-mean-token-sum': (gated_kl * mask).sum(-1).mean(),
            'seq-mean-token-mean': ((gated_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean(),
        }

        for mode, value in expected.items():
            loss, _ = compute_sdar_loss(student, teacher, mask, loss_agg_mode=mode)
            torch.testing.assert_close(loss, value, msg=mode)

    def test_unknown_aggregation_mode_raises(self):
        student, teacher, mask = _make_sdar_batch()
        with pytest.raises(ValueError, match='loss_agg_mode'):
            compute_sdar_loss(student, teacher, mask, loss_agg_mode='nope')

    def test_extreme_gap_stays_finite(self):
        student = torch.zeros(1, 2)
        teacher = torch.tensor([[50.0, -50.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        loss, metrics = compute_sdar_loss(student, teacher, mask)

        assert torch.isfinite(loss)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


class TestSDARInGRPO:

    def _batch(self):
        labels = torch.tensor([[1, 1, 1], [1, 1, 1]])
        logps = torch.tensor([[-0.5, -1.0, -0.8], [-0.3, -0.6, -0.9]])
        old_logps = logps.detach() + 0.05
        advantages = torch.tensor([[1.0], [-1.0]])
        teacher = torch.tensor([[-0.2, -1.2, -0.4], [-0.5, -0.4, -1.1]])
        return {'labels': labels}, {'logps': logps}, old_logps, advantages, teacher

    def test_auxiliary_term_is_added_with_its_coefficient(self):
        inputs, outputs, old_logps, advantages, teacher = self._batch()
        mask = inputs['labels'] != -100

        plain = GRPOLoss(epsilon=0.2)(inputs, outputs, old_logps=old_logps, advantages=advantages)
        hybrid = GRPOLoss(epsilon=0.2, sdar_loss_coef=0.5)(
            inputs, outputs, old_logps=old_logps, advantages=advantages, teacher_logps=teacher)

        unit, _ = compute_sdar_loss(outputs['logps'], teacher, mask, gate_beta=5.0)
        torch.testing.assert_close(hybrid['loss'], plain['loss'] + 0.5 * unit)

    def test_zero_coefficient_matches_plain_grpo(self):
        inputs, outputs, old_logps, advantages, teacher = self._batch()

        plain = GRPOLoss(epsilon=0.2)(inputs, outputs, old_logps=old_logps, advantages=advantages)
        no_aux = GRPOLoss(epsilon=0.2, sdar_loss_coef=0.0)(
            inputs, outputs, old_logps=old_logps, advantages=advantages, teacher_logps=teacher)

        torch.testing.assert_close(no_aux['loss'], plain['loss'])

    def test_missing_teacher_matches_plain_grpo(self):
        inputs, outputs, old_logps, advantages, _ = self._batch()

        plain = GRPOLoss(epsilon=0.2)(inputs, outputs, old_logps=old_logps, advantages=advantages)
        no_teacher = GRPOLoss(epsilon=0.2, sdar_loss_coef=0.5)(
            inputs, outputs, old_logps=old_logps, advantages=advantages)

        torch.testing.assert_close(no_teacher['loss'], plain['loss'])

    def test_coefficient_toggles_the_logits_requirement(self):
        assert GRPOLoss(sdar_loss_coef=0.5).require_logits is True
        assert GRPOLoss().require_logits is False

    def test_auxiliary_path_reports_zero_num_tokens(self):
        inputs, outputs, old_logps, advantages, teacher = self._batch()
        result = GRPOLoss(epsilon=0.2, sdar_loss_coef=0.5)(
            inputs, outputs, old_logps=old_logps, advantages=advantages, teacher_logps=teacher)
        assert result['num_tokens'] == 0
