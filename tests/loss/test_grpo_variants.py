# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the GRPO-family variants that reshape the surrogate beyond PPO clipping.

Covers GPG, DRO, DPPO-TV, DPPO-KL, GeoMean and REAL, plus numeric assertions for the
variants that previously had only a finite-loss smoke test (GSPO, SAPO, CISPO, DR-GRPO).
"""
import math

import pytest
import torch

from twinkle.loss import (
    CISPOLoss,
    DPPOKLLoss,
    DPPOTVLoss,
    DROLoss,
    DRGRPOLoss,
    GeoMeanLoss,
    GPGLoss,
    GRPOLoss,
    GSPOLoss,
    REALLoss,
    SAPOLoss,
    torch_loss_mapping,
)


def _k3_kl(logps, ref_logps, loss_mask):
    """k3 reference-KL term used by every variant's optional ``beta`` penalty."""
    delta = ref_logps - logps
    return torch.exp(delta) - delta - 1


def _seq_mean(per_token_loss, loss_mask):
    """The GRPOLoss default aggregation: token-mean per sequence, then batch mean."""
    return ((per_token_loss * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()


class TestGPGLoss:

    def test_matches_reinforce_formula(self):
        labels = torch.tensor([[1, 1, -100], [1, 1, -100]])
        logps = torch.tensor([[-0.5, -0.5, 0.0], [-1.0, -1.0, 0.0]])
        advantages = torch.tensor([[2.0], [3.0]])

        result = GPGLoss()({'labels': labels}, {'logps': logps}, advantages=advantages)

        # Row 0: -(-0.5) * 2 = 1.0 per token. Row 1: -(-1.0) * 3 = 3.0 per token.
        assert result['loss'].item() == pytest.approx(2.0)

    def test_ignores_the_importance_ratio(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])
        advantages = torch.tensor([[1.0]])

        loss_fn = GPGLoss()
        on_policy = loss_fn({'labels': labels}, {'logps': logps}, old_logps=logps, advantages=advantages)
        off_policy = loss_fn({'labels': labels}, {'logps': logps}, old_logps=logps - 1.0, advantages=advantages)

        torch.testing.assert_close(off_policy['loss'], on_policy['loss'])
        assert on_policy['loss'].item() == pytest.approx(0.5)

    def test_differs_from_clipped_grpo_on_the_same_batch(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])
        old_logps = torch.tensor([[-1.5, -1.5]])  # ratio = e > 1 + epsilon
        advantages = torch.tensor([[1.0]])

        gpg = GPGLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        grpo = GRPOLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # GRPO clips the ratio at 1.2 -> -1.2 per token; GPG stays at -logp * adv = 0.5.
        assert grpo['loss'].item() == pytest.approx(-1.2)
        assert gpg['loss'].item() == pytest.approx(0.5)

    def test_zero_advantage_gives_zero(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])
        result = GPGLoss()({'labels': labels}, {'logps': logps}, advantages=torch.zeros(1, 1))
        assert result['loss'].item() == pytest.approx(0.0)

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('gpg') is GPGLoss


class TestDROLoss:

    def test_matches_quadratic_log_ratio_formula(self):
        labels = torch.tensor([[1, 1, -100]])
        logps = torch.tensor([[-0.6, -0.4, 0.0]])
        old_logps = torch.tensor([[-0.4, -0.2, 0.0]])  # log ratio = -0.2
        advantages = torch.tensor([[1.0]])

        result = DROLoss(dro_beta=0.5)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # per_token = -(logp * adv - 0.5 * beta * log_ratio^2)
        expected_per_token = -((-0.6) * 1.0 - 0.5 * 0.5 * 0.04)
        expected_per_token_next = -((-0.4) * 1.0 - 0.5 * 0.5 * 0.04)
        assert result['loss'].item() == pytest.approx((expected_per_token + expected_per_token_next) / 2)

    def test_quadratic_penalty_is_symmetric_in_the_log_ratio(self):
        loss_fn = DROLoss(dro_beta=0.2)
        labels = torch.tensor([[1]])
        logps = torch.tensor([[-0.5]])
        advantages = torch.zeros(1, 1)

        up = loss_fn({'labels': labels}, {'logps': logps}, old_logps=logps - 0.3, advantages=advantages)
        down = loss_fn({'labels': labels}, {'logps': logps}, old_logps=logps + 0.3, advantages=advantages)

        torch.testing.assert_close(up['loss'], down['loss'])
        assert up['loss'].item() == pytest.approx(0.5 * 0.2 * 0.09)

    def test_penalty_uses_the_unclamped_log_ratio(self):
        """The base GRPO path clamps the log ratio at ±5; DRO's penalty must not."""
        labels = torch.tensor([[1]])
        logps = torch.tensor([[-0.5]])
        old_logps = torch.tensor([[9.5]])  # log ratio = -10, beyond the base ±5 clamp
        result = DROLoss(dro_beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=torch.zeros(1, 1))

        # 0.5 * 0.1 * 100 = 5.0 (a ±5-clamped ratio would give 1.25).
        assert result['loss'].item() == pytest.approx(5.0)

    def test_rejects_non_positive_beta(self):
        with pytest.raises(ValueError, match='dro_beta'):
            DROLoss(dro_beta=0.0)

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -1.0]])
        old_logps = torch.tensor([[-0.6, -1.1]])
        ref_logps = torch.tensor([[-0.7, -0.8]])
        advantages = torch.tensor([[1.0]])
        mask = labels != -100

        plain = DROLoss(dro_beta=0.5)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        with_kl = DROLoss(dro_beta=0.5, beta=0.01)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, ref_logps=ref_logps,
            advantages=advantages)

        expected = plain['loss'] + 0.01 * _seq_mean(_k3_kl(logps, ref_logps, mask), mask)
        torch.testing.assert_close(with_kl['loss'], expected)

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('dro') is DROLoss


class TestDPPOTVLoss:

    def test_masks_positive_advantage_tokens_beyond_the_tv_threshold(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[0.0, -math.log(2)]])
        old_logps = torch.tensor([[-math.log(2), -math.log(2)]])
        advantages = torch.tensor([[1.0]])

        result = DPPOTVLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                         advantages=advantages)

        # Token 0: prob - old_prob = 0.5 > epsilon_high -> invalid. Token 1 survives with ratio 1.
        assert result['loss'].item() == pytest.approx(math.log(2) / 2)

    def test_negative_advantage_uses_the_low_divergence_branch(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[0.0, -math.log(2)]])
        old_logps = torch.tensor([[-math.log(2), -math.log(2)]])
        advantages = torch.tensor([[-1.0]])

        result = DPPOTVLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                         advantages=advantages)

        # Token 0 is now valid (0.5 >= -epsilon). Token 1 contributes +log(2); token 0 contributes 0.
        assert result['loss'].item() == pytest.approx(-math.log(2) / 2)

    def test_truncates_the_ratio_at_clip_ratio_c(self):
        labels = torch.tensor([[1]])
        logps = torch.tensor([[-5.0]])
        old_logps = torch.tensor([[-10.0]])  # ratio = exp(5) ~= 148, truncated to 20
        advantages = torch.tensor([[1.0]])

        result = DPPOTVLoss(epsilon=0.2, clip_ratio_c=20.0)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        assert result['loss'].item() == pytest.approx(100.0)

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-1.0, -2.0]])
        old_logps = logps.clone()
        ref_logps = torch.tensor([[-1.3, -1.8]])
        advantages = torch.tensor([[1.0]])
        mask = labels != -100

        plain = DPPOTVLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        with_kl = DPPOTVLoss(beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, ref_logps=ref_logps,
            advantages=advantages)

        torch.testing.assert_close(with_kl['loss'], plain['loss'] + 0.1 * _seq_mean(_k3_kl(logps, ref_logps, mask),
                                                                                   mask))

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('dppo_tv') is DPPOTVLoss


class TestDPPOKLLoss:

    def test_masks_high_binary_kl_for_positive_advantage(self):
        labels = torch.tensor([[1]])
        logps = torch.tensor([[math.log(0.9)]])
        old_logps = torch.tensor([[math.log(0.3)]])
        advantages = torch.tensor([[1.0]])

        result = DPPOKLLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                         advantages=advantages)

        assert result['loss'].item() == pytest.approx(0.0)

    def test_probability_decrease_stays_valid_via_the_or_condition(self):
        labels = torch.tensor([[1]])
        logps = torch.tensor([[math.log(0.3)]])
        old_logps = torch.tensor([[math.log(0.9)]])
        advantages = torch.tensor([[1.0]])

        result = DPPOKLLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                         advantages=advantages)

        # binary_kl is large but prob <= old_prob keeps the positive branch valid.
        expected = -(1.0) * (0.3 / 0.9) * math.log(0.3)
        assert result['loss'].item() == pytest.approx(expected)

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-1.0, -2.0]])
        old_logps = logps.clone()
        ref_logps = torch.tensor([[-1.3, -1.8]])
        advantages = torch.tensor([[1.0]])
        mask = labels != -100

        plain = DPPOKLLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        with_kl = DPPOKLLoss(beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, ref_logps=ref_logps,
            advantages=advantages)

        torch.testing.assert_close(with_kl['loss'], plain['loss'] + 0.1 * _seq_mean(_k3_kl(logps, ref_logps, mask),
                                                                                   mask))

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('dppo_kl') is DPPOKLLoss


class TestGeoMeanLoss:

    def test_unclipped_ratios_reduce_to_the_sequence_advantage(self):
        labels = torch.tensor([[1, 1, -100]])
        logps = torch.tensor([[-0.2, -0.3, 0.0]])
        old_logps = torch.tensor([[-0.3, -0.2, 0.0]])  # log ratios +0.1 / -0.1 cancel in the mean
        advantages = torch.tensor([[2.0]])

        result = GeoMeanLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                               advantages=advantages)

        assert result['loss'].item() == pytest.approx(-2.0)

    def test_positive_advantage_clips_the_log_ratio_pessimistically(self):
        labels = torch.tensor([[1, 1, -100]])
        logps = torch.tensor([[-0.5, -0.5, 0.0]])
        old_logps = torch.tensor([[-1.0, -1.0, 0.0]])  # log ratio 0.5 > epsilon = 0.2
        advantages = torch.tensor([[1.0]])

        result = GeoMeanLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                          advantages=advantages)

        assert result['loss'].item() == pytest.approx(-math.exp(0.2))

    def test_masked_positions_are_excluded_from_the_geometric_mean(self):
        labels = torch.tensor([[1, 1, -100]])
        logps = torch.tensor([[-0.5, -0.5, 0.0]])
        old_logps = torch.tensor([[-1.0, -1.0, 0.0]])
        advantages = torch.tensor([[1.0]])

        baseline = GeoMeanLoss(epsilon=0.2)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        wild = logps.clone()
        wild_old = old_logps.clone()
        wild[0, 2] = 100.0
        wild_old[0, 2] = -100.0
        perturbed = GeoMeanLoss(epsilon=0.2)(
            {'labels': labels}, {'logps': wild}, old_logps=wild_old, advantages=advantages)

        torch.testing.assert_close(perturbed['loss'], baseline['loss'])

    def test_zero_advantage_gives_zero(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -1.0]])
        result = GeoMeanLoss()({'labels': labels}, {'logps': logps}, old_logps=logps,
                               advantages=torch.zeros(1, 1))
        assert result['loss'].item() == pytest.approx(0.0)

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -1.0]])
        old_logps = logps.clone()
        ref_logps = torch.tensor([[-0.7, -0.8]])
        advantages = torch.tensor([[1.0]])
        mask = labels != -100

        plain = GeoMeanLoss()({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                              advantages=advantages)
        with_kl = GeoMeanLoss(beta=0.1)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                       ref_logps=ref_logps, advantages=advantages)

        torch.testing.assert_close(with_kl['loss'], plain['loss'] + 0.1 * _seq_mean(_k3_kl(logps, ref_logps, mask),
                                                                                   mask))

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('geo_mean') is GeoMeanLoss


class TestREALLoss:

    def test_matches_the_group_soft_constraint_closed_form(self):
        labels = torch.tensor([[1, 1], [1, 1]])
        logps = torch.zeros(2, 2)
        advantages = torch.tensor([[1.0], [-1.0]])

        result = REALLoss(real_tau=0.5, num_generations=2)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, advantages=advantages)

        # One balanced group with both scores at 0: log(1+e^0) for each side.
        assert result['loss'].item() == pytest.approx(2 * math.log(2))

    def test_all_positive_group_is_skipped_but_stays_connected(self):
        labels = torch.tensor([[1, 1], [1, 1]])
        logps = torch.zeros(2, 2, requires_grad=True)
        advantages = torch.tensor([[1.0], [2.0]])

        result = REALLoss(real_tau=0.5, num_generations=2)(
            {'labels': labels}, {'logps': logps}, old_logps=logps.detach(), advantages=advantages)

        assert result['loss'].item() == pytest.approx(0.0)
        result['loss'].backward()
        assert logps.grad is not None

    def test_rejects_batch_not_divisible_by_num_generations(self):
        labels = torch.tensor([[1], [1], [1]])
        logps = torch.zeros(3, 1)
        advantages = torch.tensor([[1.0], [-1.0], [1.0]])

        with pytest.raises(AssertionError, match='num_generations'):
            REALLoss(num_generations=2)({'labels': labels}, {'logps': logps}, old_logps=logps,
                                        advantages=advantages)

    def test_uses_the_unclamped_log_ratio(self):
        labels = torch.tensor([[1], [1]])
        logps = torch.tensor([[10.0], [-10.0]])
        old_logps = torch.zeros(2, 1)  # log ratios +10 / -10, beyond the base ±5 clamp
        advantages = torch.tensor([[1.0], [-1.0]])

        result = REALLoss(real_tau=0.5, num_generations=2)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # Scores +10 / -10 -> scaled ±20, so both soft constraints are ~e^-20. A ±5 clamp would
        # instead leave a ~9e-5 loss.
        assert result['loss'].item() < 1e-6

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1], [1, 1]])
        logps = torch.zeros(2, 2)
        ref_logps = torch.full((2, 2), -0.4)
        advantages = torch.tensor([[1.0], [-1.0]])
        mask = labels != -100

        plain = REALLoss(real_tau=0.5, num_generations=2)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, advantages=advantages)
        with_kl = REALLoss(real_tau=0.5, num_generations=2, beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, ref_logps=ref_logps,
            advantages=advantages)

        expected_kl = (_k3_kl(logps, ref_logps, mask) * mask).sum() / mask.sum()
        torch.testing.assert_close(with_kl['loss'], plain['loss'] + 0.1 * expected_kl)

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('real') is REALLoss


class TestThinVariantCoverage:
    """Numeric assertions for variants that previously only asserted ``isfinite``."""

    def test_gspo_uses_the_sequence_level_ratio(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.1, -0.1]])
        old_logps = torch.tensor([[-0.2, -0.2]])  # per-token log ratio 0.1 -> sequence ratio e^0.1
        advantages = torch.tensor([[1.0]])

        result = GSPOLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=old_logps,
                                      advantages=advantages)

        assert result['loss'].item() == pytest.approx(-math.exp(0.1))

    def test_sapo_soft_gate_at_unit_ratio(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])

        positive = SAPOLoss(epsilon=0.2, tau_pos=1.0, tau_neg=1.0)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, advantages=torch.tensor([[1.0]]))
        negative = SAPOLoss(epsilon=0.2, tau_pos=1.0, tau_neg=1.0)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, advantages=torch.tensor([[-1.0]]))

        # sigmoid(0) * 4 / tau = 2 at ratio 1, so the loss is -2 * advantage.
        assert positive['loss'].item() == pytest.approx(-2.0)
        assert negative['loss'].item() == pytest.approx(2.0)

    def test_cispo_clamped_ratio_times_logp(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])
        advantages = torch.tensor([[1.0]])

        result = CISPOLoss(epsilon=0.2)({'labels': labels}, {'logps': logps}, old_logps=logps,
                                       advantages=advantages)

        # ratio = 1 (not clamped), per-token = -1 * 1 * (-0.5) = 0.5, token-mean = 0.5.
        assert result['loss'].item() == pytest.approx(0.5)

    def test_dr_grpo_normalizes_by_batch_times_max_completion_length(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-0.5, -0.5]])
        advantages = torch.tensor([[1.0]])

        result = DRGRPOLoss(epsilon=0.2, max_completion_length=4)(
            {'labels': labels}, {'logps': logps}, old_logps=logps, advantages=advantages)

        # Two tokens at -1.0 each -> -2.0 / (batch 1 * max_completion_length 4) = -0.5.
        assert result['loss'].item() == pytest.approx(-0.5)
