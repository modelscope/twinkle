# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for MOPDLoss -- multi-teacher on-policy distillation fused into one mixture target."""
import pytest
import torch

from twinkle.loss import MOPDLoss, OPSDLoss, torch_loss_mapping


def _make_batch(batch_size=2, seq_len=6):
    labels = torch.full((batch_size, seq_len), 1)
    labels[:, seq_len // 2:] = -100
    logps = torch.full((batch_size, seq_len), -0.5)
    logps[:, seq_len // 2:] = 0.0
    return {'labels': labels}, {'logps': logps}


def _teacher_tensors(logps):
    """Two teachers: one more confident than the student, one less confident."""
    return logps - 0.4, logps + 0.2


class TestMOPDLoss:

    def test_single_teacher_degenerates_to_opsd(self):
        inputs, outputs = _make_batch()
        teacher = outputs['logps'] - 0.4

        mopd = MOPDLoss()(inputs, outputs, teacher_logps=[teacher])
        opsd = OPSDLoss()(inputs, outputs, teacher_logps=teacher)

        torch.testing.assert_close(mopd['loss'], opsd['loss'])

    def test_two_teacher_probability_mixture_matches_closed_form(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        teacher1, teacher2 = _teacher_tensors(logps)
        weights = [0.3, 0.7]

        result = MOPDLoss()(inputs, outputs, teacher_logps=[teacher1, teacher2], teacher_weights=weights)

        mix = torch.log(weights[0] * torch.exp(teacher1) + weights[1] * torch.exp(teacher2))
        r = torch.clamp(mix - logps, min=-10.0, max=10.0)
        per_token = torch.exp(r) - r - 1
        mask = inputs['labels'] != -100
        expected = (per_token * mask).sum() / mask.sum()

        torch.testing.assert_close(result['loss'], expected)

    def test_default_weights_are_uniform(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        teachers = [logps - 0.4, logps + 0.2, logps - 0.1]

        default = MOPDLoss()(inputs, outputs, teacher_logps=teachers)
        explicit = MOPDLoss()(inputs, outputs, teacher_logps=teachers,
                              teacher_weights=[1.0 / 3.0] * 3)

        torch.testing.assert_close(default['loss'], explicit['loss'])

    def test_unnormalised_weights_are_normalised(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        teachers = [logps - 0.4, logps + 0.2]

        unnormalised = MOPDLoss()(inputs, outputs, teacher_logps=teachers, teacher_weights=[1.0, 3.0])
        normalised = MOPDLoss()(inputs, outputs, teacher_logps=teachers, teacher_weights=[0.25, 0.75])

        torch.testing.assert_close(unnormalised['loss'], normalised['loss'])

    def test_zero_weight_teacher_is_skipped(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        teacher1 = logps - 0.4
        dead_teacher = torch.full_like(logps, -1e30)

        result = MOPDLoss()(inputs, outputs, teacher_logps=[teacher1, dead_teacher],
                            teacher_weights=[1.0, 0.0])
        single = MOPDLoss()(inputs, outputs, teacher_logps=[teacher1], teacher_weights=[1.0])

        assert torch.isfinite(result['loss'])
        torch.testing.assert_close(result['loss'], single['loss'])

    def test_weight_length_must_match_teacher_count(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        with pytest.raises(ValueError, match='one-to-one'):
            MOPDLoss()(inputs, outputs, teacher_logps=[logps, logps], teacher_weights=[1.0])

    def test_negative_weight_rejected(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        with pytest.raises(ValueError, match='non-negative'):
            MOPDLoss()(inputs, outputs, teacher_logps=[logps, logps], teacher_weights=[1.0, -0.5])

    def test_zero_sum_weights_rejected(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        with pytest.raises(ValueError, match='positive finite'):
            MOPDLoss()(inputs, outputs, teacher_logps=[logps, logps], teacher_weights=[0.0, 0.0])

    def test_bare_tensor_teacher_delegates_to_opsd(self):
        inputs, outputs = _make_batch()
        teacher = outputs['logps'] - 0.3

        mopd = MOPDLoss()(inputs, outputs, teacher_logps=teacher)
        opsd = OPSDLoss()(inputs, outputs, teacher_logps=teacher)

        torch.testing.assert_close(mopd['loss'], opsd['loss'])

    def test_ragged_multi_teacher_matches_dense(self):
        inputs, outputs = _make_batch()
        logps = outputs['logps']
        teacher1, teacher2 = _teacher_tensors(logps)
        mask = inputs['labels'] != -100

        dense = MOPDLoss()(inputs, outputs, teacher_logps=[teacher1, teacher2], teacher_weights=[0.4, 0.6])
        ragged = MOPDLoss()(inputs, outputs,
                            teacher_logps=[[teacher1[i][mask[i]].tolist() for i in range(teacher1.shape[0])],
                                           [teacher2[i][mask[i]].tolist() for i in range(teacher2.shape[0])]],
                            teacher_weights=[0.4, 0.6])

        torch.testing.assert_close(ragged['loss'], dense['loss'])

    def test_missing_teacher_returns_zero_that_flows(self):
        labels = torch.tensor([[1, 1, -100]])
        logps = torch.full((1, 3), -0.5, requires_grad=True)

        result = MOPDLoss()({'labels': labels}, {'logps': logps})

        assert result['loss'].item() == pytest.approx(0.0)
        result['loss'].backward()
        assert logps.grad is not None

    def test_registered_and_requires_logps(self):
        assert torch_loss_mapping.get('mopd') is MOPDLoss
        assert MOPDLoss.require_logps is True
        assert MOPDLoss.require_logits is False
