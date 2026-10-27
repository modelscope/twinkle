# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for KTOLoss -- the unpaired (single-completion + binary label) preference objective."""
import pytest
import torch

from twinkle.loss import KTOLoss, torch_loss_mapping


def _make_kto_batch(batch_size=2, seq_len=4):
    """Policy/ref logps chosen so every valid token has logratio = policy - ref = 0.2."""
    labels = torch.full((batch_size, seq_len), 1)
    labels[:, seq_len // 2:] = -100
    logps = torch.full((batch_size, seq_len), -0.5)
    ref_logps = torch.full((batch_size, seq_len), -0.7)
    # Ignored positions are never scored, so their values must not matter.
    logps[:, seq_len // 2:] = 0.0
    ref_logps[:, seq_len // 2:] = 0.0
    return {'labels': labels}, {'logps': logps}, ref_logps


def _sequence_logratios(inputs, outputs, ref_logps):
    """Independent oracle: per-example sum of valid-token (policy - ref) log-probs."""
    mask = (inputs['labels'] != -100).float()
    policy = (outputs['logps'] * mask).sum(-1)
    reference = (ref_logps * mask).sum(-1)
    return policy - reference


class TestKTOLoss:

    def test_matches_manual_formula(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        result = KTOLoss(beta=0.1)(inputs, outputs, label=[True, False], z_kl=0.0, ref_logps=ref_logps)

        logratios = _sequence_logratios(inputs, outputs, ref_logps)
        desirable = 1 - torch.sigmoid(0.1 * (logratios[0] - 0.0))
        undesirable = 1 - torch.sigmoid(0.1 * (0.0 - logratios[1]))
        expected = (desirable + undesirable) / 2

        torch.testing.assert_close(result['loss'], expected)

    def test_z_kl_shifts_the_anchor(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        z_kl = 0.5
        result = KTOLoss(beta=0.1)(inputs, outputs, label=[True, False], z_kl=z_kl, ref_logps=ref_logps)

        logratios = _sequence_logratios(inputs, outputs, ref_logps)
        desirable = 1 - torch.sigmoid(0.1 * (logratios[0] - z_kl))
        undesirable = 1 - torch.sigmoid(0.1 * (z_kl - logratios[1]))
        torch.testing.assert_close(result['loss'], (desirable + undesirable) / 2)

    def test_asymmetric_weights_scale_each_side(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        result = KTOLoss(beta=0.1, desirable_weight=2.0, undesirable_weight=0.5)(
            inputs, outputs, label=[True, False], ref_logps=ref_logps)

        logratios = _sequence_logratios(inputs, outputs, ref_logps)
        desirable = 2.0 * (1 - torch.sigmoid(0.1 * logratios[0]))
        undesirable = 0.5 * (1 - torch.sigmoid(-0.1 * logratios[1]))
        torch.testing.assert_close(result['loss'], (desirable + undesirable) / 2)

    def test_all_desirable_batch(self):
        inputs, outputs, ref_logps = _make_kto_batch(batch_size=1)
        result = KTOLoss(beta=0.1)(inputs, outputs, label=[True], ref_logps=ref_logps)

        logratio = _sequence_logratios(inputs, outputs, ref_logps)[0]
        torch.testing.assert_close(result['loss'], 1 - torch.sigmoid(0.1 * logratio))

    def test_all_undesirable_batch(self):
        inputs, outputs, ref_logps = _make_kto_batch(batch_size=1)
        result = KTOLoss(beta=0.1)(inputs, outputs, label=[False], ref_logps=ref_logps)

        logratio = _sequence_logratios(inputs, outputs, ref_logps)[0]
        torch.testing.assert_close(result['loss'], 1 - torch.sigmoid(-0.1 * logratio))

    def test_missing_label_raises(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        with pytest.raises(ValueError, match='label'):
            KTOLoss()(inputs, outputs, ref_logps=ref_logps)

    def test_missing_ref_logps_raises(self):
        inputs, outputs, _ = _make_kto_batch()
        with pytest.raises(ValueError, match='reference logps'):
            KTOLoss()(inputs, outputs, label=[True, False])

    def test_label_count_mismatch_raises(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        with pytest.raises(ValueError, match='label count'):
            KTOLoss()(inputs, outputs, label=[True, False, True], ref_logps=ref_logps)

    def test_gradient_pushes_desirable_up_and_leaves_masked_alone(self):
        labels = torch.tensor([[1, 1, -100, -100]])
        logps = torch.full((1, 4), -0.5, requires_grad=True)
        ref_logps = torch.full((1, 4), -0.7)

        result = KTOLoss(beta=0.1)({'labels': labels}, {'logps': logps}, label=[True], ref_logps=ref_logps)
        result['loss'].backward()

        # Desirable below its anchor: loss falls as the policy log-prob rises -> negative gradient.
        assert (logps.grad[0, :2] < 0).all()
        assert torch.allclose(logps.grad[0, 2:], torch.zeros(2))

    def test_ragged_ref_logps_match_dense(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        mask = inputs['labels'] != -100
        ragged = [ref_logps[i][mask[i]].tolist() for i in range(ref_logps.shape[0])]

        loss_fn = KTOLoss(beta=0.1)
        dense_result = loss_fn(inputs, outputs, label=[True, False], ref_logps=ref_logps)
        ragged_result = loss_fn(inputs, outputs, label=[True, False], ref_logps=ragged)

        torch.testing.assert_close(ragged_result['loss'], dense_result['loss'])

    def test_reports_zero_num_tokens(self):
        inputs, outputs, ref_logps = _make_kto_batch()
        result = KTOLoss()(inputs, outputs, label=[True, False], ref_logps=ref_logps)
        assert result['num_tokens'] == 0

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('kto') is KTOLoss
