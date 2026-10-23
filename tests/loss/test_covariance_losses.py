# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the covariance-guided PPO variants: ClipCov (token zeroing) and KLCov (token KL)."""
import math

import pytest
import torch

from twinkle.loss import ClipCovLoss, KLCovLoss, torch_loss_mapping


def _manual_ppo(logps, old_logps, advantages, loss_mask, zero_mask=None, eps=0.2, eps_high=0.2):
    """Reference PPO clipped surrogate with an optional per-token zeroing mask."""
    ratio = torch.exp(logps - old_logps)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - eps, 1 + eps_high)
    per_token = torch.maximum(pg_losses1, pg_losses2)
    if zero_mask is not None:
        per_token = per_token * zero_mask
    return ((per_token * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()


def _k3_kl(logps, ref_logps):
    delta = ref_logps - logps
    return torch.exp(delta) - delta - 1


def _seq_mean(values, loss_mask):
    return ((values * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)).mean()


class TestClipCovLoss:

    def test_rejects_non_positive_ratio(self):
        with pytest.raises(ValueError, match='clip_cov_ratio'):
            ClipCovLoss(clip_cov_ratio=0.0)

    def test_empty_covariance_window_matches_plain_ppo(self):
        torch.manual_seed(0)
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-0.3, -0.4, -0.6]])
        old_logps = torch.tensor([[-0.4, -0.4, -0.4]])
        advantages = torch.tensor([[1.0, -1.0, 0.5]])
        mask = labels != -100

        result = ClipCovLoss(clip_cov_lb=100.0, clip_cov_ub=200.0)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        torch.testing.assert_close(result['loss'],
                                   _manual_ppo(logps, old_logps, advantages, mask))

    def test_zeroes_the_only_token_inside_the_covariance_window(self):
        torch.manual_seed(0)
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-0.3, -0.4, -0.6]])
        old_logps = torch.tensor([[-0.4, -0.4, -0.4]])
        # Only token 0 carries advantage, and its covariance (~0.089) is the sole value in (0.07, 0.1).
        advantages = torch.tensor([[1.0, 0.0, 0.0]])
        mask = labels != -100

        result = ClipCovLoss(clip_cov_ratio=0.5, clip_cov_lb=0.07, clip_cov_ub=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        assert result['loss'].item() == pytest.approx(0.0)
        assert _manual_ppo(logps, old_logps, advantages, mask).item() < -0.1

    def test_already_clipped_tokens_are_excluded_from_selection(self):
        torch.manual_seed(0)
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[0.5, -0.4, -0.5]])
        old_logps = torch.tensor([[-0.4, -0.4, -0.4]])
        # Token 0 has the largest covariance (~0.42) but is also PPO-clipped, so it must NOT be zeroed.
        # Token 2 (~0.12) is the only candidate in the (0.1, 0.2) window and is the one zeroed.
        advantages = torch.tensor([[2.0, 1.0, 1.0]])
        mask = labels != -100

        result = ClipCovLoss(clip_cov_ratio=0.5, clip_cov_lb=0.1, clip_cov_ub=0.2)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # -2.4 (clipped token 0) + -1.0 (token 1) + 0 (zeroed token 2), over 3 tokens.
        assert result['loss'].item() == pytest.approx(-3.4 / 3)

    def test_reference_kl_is_added_after_zeroing(self):
        torch.manual_seed(0)
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-0.3, -0.4, -0.6]])
        old_logps = torch.tensor([[-0.4, -0.4, -0.4]])
        ref_logps = torch.tensor([[-0.5, -0.5, -0.5]])
        advantages = torch.tensor([[1.0, 0.0, 0.0]])
        mask = labels != -100

        plain = ClipCovLoss(clip_cov_ratio=0.5, clip_cov_lb=0.07, clip_cov_ub=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        with_kl = ClipCovLoss(clip_cov_ratio=0.5, clip_cov_lb=0.07, clip_cov_ub=0.1, beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, ref_logps=ref_logps,
            advantages=advantages)

        assert plain['loss'].item() == pytest.approx(0.0)
        torch.testing.assert_close(with_kl['loss'],
                                   plain['loss'] + 0.1 * _seq_mean(_k3_kl(logps, ref_logps), mask))

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('clip_cov') is ClipCovLoss


class TestKLCovLoss:

    def test_rejects_non_positive_ratio(self):
        with pytest.raises(ValueError, match='kl_cov_ratio'):
            KLCovLoss(kl_cov_ratio=0.0)

    def test_ratio_one_penalises_every_valid_token(self):
        labels = torch.tensor([[1, 1]])
        logps = torch.tensor([[-1.0, -2.0]])
        old_logps = torch.tensor([[-2.0, -2.0]])
        advantages = torch.tensor([[1.0, 1.0]])

        result = KLCovLoss(kl_cov_ratio=1.0, ppo_kl_coef=1.0)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # token 0: -e + 1, token 1: -1 + 0 -> mean = -e / 2.
        assert result['loss'].item() == pytest.approx(-math.e / 2)

    def test_penalty_applies_only_to_the_highest_covariance_token(self):
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-1.0, -2.0, -5.0]])
        old_logps = torch.tensor([[-2.0, -2.0, -2.0]])
        # Covariances (computed over the valid tokens): [5/3, 0, 7/3] -> token 2 wins.
        advantages = torch.tensor([[2.0, 1.0, 0.0]])

        result = KLCovLoss(kl_cov_ratio=0.5, ppo_kl_coef=1.0)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        # pg = [-2e, -1, 0]; only token 2 is selected and becomes pg + 1 * |log_ratio| = 3.
        assert result['loss'].item() == pytest.approx((2 - 2 * math.e) / 3)

    def test_zero_coefficient_matches_unclipped_ppo(self):
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-1.0, -2.0, -5.0]])
        old_logps = torch.tensor([[-2.0, -2.0, -2.0]])
        advantages = torch.tensor([[2.0, 1.0, 0.0]])
        mask = labels != -100

        result = KLCovLoss(kl_cov_ratio=0.5, ppo_kl_coef=0.0)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)

        expected = _seq_mean(-advantages * torch.exp(logps - old_logps), mask)
        torch.testing.assert_close(result['loss'], expected)

    def test_adds_reference_kl_penalty(self):
        labels = torch.tensor([[1, 1, 1]])
        logps = torch.tensor([[-1.0, -2.0, -5.0]])
        old_logps = torch.tensor([[-2.0, -2.0, -2.0]])
        ref_logps = torch.tensor([[-2.5, -1.5, -4.0]])
        advantages = torch.tensor([[2.0, 1.0, 0.0]])
        mask = labels != -100

        plain = KLCovLoss(kl_cov_ratio=0.5)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, advantages=advantages)
        with_kl = KLCovLoss(kl_cov_ratio=0.5, beta=0.1)(
            {'labels': labels}, {'logps': logps}, old_logps=old_logps, ref_logps=ref_logps,
            advantages=advantages)

        torch.testing.assert_close(with_kl['loss'],
                                   plain['loss'] + 0.1 * _seq_mean(_k3_kl(logps, ref_logps), mask))

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('kl_cov') is KLCovLoss
