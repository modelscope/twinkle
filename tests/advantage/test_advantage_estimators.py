# Advantage estimator tests (plan T2-01..T2-19): the six estimators not covered by
# test_advantage.py (GPG / OPO / GRPOPassK / ReinforcePlusPlus / ReinforcePPBaseline / ReMax)
# plus the teacher_signal shaping functions.
import math

import pytest
import torch

from twinkle.advantage import (GPGAdvantage, GRPOPassKAdvantage, OPOAdvantage, ReMaxAdvantage,
                                ReinforcePPBaselineAdvantage, ReinforcePlusPlusAdvantage)
from twinkle.advantage.teacher_signal import (apply_rlsd_reweight, compute_teacher_logratio,
                                              expand_advantage_to_per_token)

ATOL = 1e-6


class TestGPGAdvantage:
    """T2-01/T2-02/T2-03: the dynamic alpha = N / count_nonzero, f_norm, group centering."""

    def test_dynamic_alpha_hand_computed(self):
        # rewards [1,0,0,0], K=4: nonzero count=1 -> alpha=4; group mean=0.25.
        adv = GPGAdvantage()([1.0, 0.0, 0.0, 0.0], num_generations=4)
        torch.testing.assert_close(adv, torch.tensor([3.0, -1.0, -1.0, -1.0]), atol=ATOL, rtol=0)

    def test_f_norm_divides(self):
        adv = GPGAdvantage()([1.0, 0.0, 0.0, 0.0], num_generations=4, f_norm=2.0)
        torch.testing.assert_close(adv, torch.tensor([1.5, -0.5, -0.5, -0.5]), atol=ATOL, rtol=0)

    def test_two_groups_centered_per_group(self):
        # [2,0 | 4,2], K=2: alpha=4/3; means=[1,3] -> 4/3 * [1,-1,1,-1].
        adv = GPGAdvantage()([2.0, 0.0, 4.0, 2.0], num_generations=2)
        torch.testing.assert_close(adv, torch.tensor([4 / 3, -4 / 3, 4 / 3, -4 / 3]), atol=ATOL, rtol=0)

    def test_all_zero_rewards_clamped_not_nan(self):
        # count_nonzero=0 -> clamp(min=1) -> alpha=N; centered zeros stay zeros.
        adv = GPGAdvantage()([0.0, 0.0, 0.0, 0.0], num_generations=2)
        torch.testing.assert_close(adv, torch.zeros(4), atol=ATOL, rtol=0)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError):
            GPGAdvantage()([1.0, 2.0, 3.0], num_generations=2)


class TestOPOAdvantage:
    """T2-04/T2-05/T2-06: length-weighted group baseline, required kwarg, divisibility."""

    def test_length_weighted_baseline_hand_computed(self):
        # lens=[1,3], scores=[2,4] -> baseline=(1*2+3*4)/4=3.5 -> adv=[-1.5, 0.5].
        adv = OPOAdvantage()([2.0, 4.0], num_generations=2, response_lengths=[1, 3])
        torch.testing.assert_close(adv, torch.tensor([-1.5, 0.5]), atol=ATOL, rtol=0)

    def test_missing_response_lengths_raises(self):
        with pytest.raises(ValueError, match='response_lengths'):
            OPOAdvantage()([2.0, 4.0], num_generations=2)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError):
            OPOAdvantage()([1.0, 2.0, 3.0], num_generations=2, response_lengths=[1, 1, 1])


class TestGRPOPassKAdvantage:
    """T2-07/T2-08/T2-09: only the best sample is non-zero, std normalization, K>=2 guard."""

    def test_only_best_sample_nonzero(self):
        # [1,3,2]: best is 3 at idx 1, gap to second-best (2) is 1.
        adv = GRPOPassKAdvantage()([1.0, 3.0, 2.0], num_generations=3, scale='none')
        torch.testing.assert_close(adv, torch.tensor([0.0, 1.0, 0.0]), atol=ATOL, rtol=0)

    def test_norm_by_group_std(self):
        # std([1,3,2]) unbiased = 1.0 -> gap 1 / (1 + eps).
        adv = GRPOPassKAdvantage()([1.0, 3.0, 2.0], num_generations=3, scale='group')
        expected = 1.0 / (1.0 + 1e-6)
        torch.testing.assert_close(adv[1], torch.tensor(expected), atol=ATOL, rtol=0)
        assert adv[0] == 0.0 and adv[2] == 0.0

    def test_k_below_two_raises(self):
        with pytest.raises(ValueError, match='>= 2'):
            GRPOPassKAdvantage()([1.0, 2.0], num_generations=1)

    def test_tied_top_two_gives_zero_advantage(self):
        # r_max == r_second_max -> the winner's advantage is exactly zero.
        adv = GRPOPassKAdvantage()([5.0, 5.0, 1.0], num_generations=3, scale='none')
        torch.testing.assert_close(adv, torch.zeros(3), atol=ATOL, rtol=0)


class TestReinforcePlusPlus:
    """T2-10/T2-11/T2-12: normalizer semantics vs GRPO, the gdpo no-op, and the K=1 branch."""

    def test_normalizer_differs_from_grpo_at_batch_scale(self):
        rewards = [0.0, 1.0, 10.0, 11.0]
        kwargs = dict(num_generations=2, scale='batch')
        rpp = ReinforcePlusPlusAdvantage()(rewards, **kwargs)
        from twinkle.advantage import GRPOAdvantage
        grpo = GRPOAdvantage()(rewards, **kwargs)
        # centered = [-.5,.5,-.5,.5]: R++ divides by the unbiased ADVANTAGE std (sqrt(1/3))
        # -> +-0.866; GRPO divides by the raw REWARD std (~5.80) -> much smaller values.
        rpp_expected = 0.5 / math.sqrt(1.0 / 3.0)
        torch.testing.assert_close(rpp.abs(), torch.full((4,), rpp_expected), atol=1e-4, rtol=0)
        assert grpo.abs().max() < 0.2

    def test_gdpo_scale_is_no_op(self):
        # scale='gdpo' leaves the centered advantage unscaled.
        adv = ReinforcePlusPlusAdvantage()([1.0, 3.0], num_generations=2, scale='gdpo')
        torch.testing.assert_close(adv, torch.tensor([-1.0, 1.0]), atol=ATOL, rtol=0)

    def test_k_equals_one_stays_zero_not_nan(self):
        adv = ReinforcePlusPlusAdvantage()([5.0], num_generations=1, scale='group')
        torch.testing.assert_close(adv, torch.zeros(1), atol=ATOL, rtol=0)


class TestReinforcePPBaseline:
    """T2-13: the masked_whiten form -- (x - mean) * rsqrt(var_unbiased + 1e-8), hand-computed."""

    def test_whitening_hand_computed(self):
        # [1,2 | 3,4], K=2 -> centered [-.5,.5,-.5,.5]; mean 0; var_unbiased = 4*.25/3 = 1/3.
        # masked_whiten = (x - mean) * rsqrt(var_unbiased + 1e-8) -> +-0.5*sqrt(3).
        adv = ReinforcePPBaselineAdvantage()([1.0, 2.0, 3.0, 4.0], num_generations=2, scale='batch')
        scale = 0.5 * math.sqrt(3.0)
        torch.testing.assert_close(adv.abs(), torch.full((4,), scale), atol=1e-4, rtol=0)

    def test_scale_none_keeps_centered(self):
        adv = ReinforcePPBaselineAdvantage()([1.0, 3.0], num_generations=2, scale='none')
        torch.testing.assert_close(adv, torch.tensor([-1.0, 1.0]), atol=ATOL, rtol=0)


class TestReMaxAdvantage:
    """T2-14/T2-15/T2-16/T2-17: required baselines, subtraction, batch scaling, K ignored."""

    def test_missing_baselines_raises(self):
        with pytest.raises(ValueError, match='reward_baselines'):
            ReMaxAdvantage()([1.0, 2.0])

    def test_subtracts_baseline(self):
        adv = ReMaxAdvantage()([3.0, 5.0], reward_baselines=[2.0, 4.0])
        torch.testing.assert_close(adv, torch.tensor([1.0, 1.0]), atol=ATOL, rtol=0)

    def test_batch_scale_hand_computed(self):
        # adv [1,3]: std = sqrt(2); normalized [1/sqrt(2), 3/sqrt(2)].
        adv = ReMaxAdvantage()([1.0, 3.0], reward_baselines=[0.0, 0.0], scale='batch')
        expected = torch.tensor([1.0, 3.0]) / (math.sqrt(2.0) + 1e-8)
        torch.testing.assert_close(adv, expected, atol=1e-4, rtol=0)

    def test_num_generations_ignored(self):
        once = ReMaxAdvantage()([3.0, 5.0], reward_baselines=[2.0, 4.0], num_generations=1)
        twice = ReMaxAdvantage()([3.0, 5.0], reward_baselines=[2.0, 4.0], num_generations=4)
        torch.testing.assert_close(once, twice, atol=0, rtol=0)


class TestListInputParity:
    """T2-19: every new estimator accepts plain lists identically to tensors."""

    @pytest.mark.parametrize('estimator,kwargs', [
        (GPGAdvantage, dict(num_generations=2)),
        (OPOAdvantage, dict(num_generations=2, response_lengths=[1, 3])),
        (GRPOPassKAdvantage, dict(num_generations=3, scale='none')),
        (ReinforcePlusPlusAdvantage, dict(num_generations=2)),
        (ReinforcePPBaselineAdvantage, dict(num_generations=2)),
        (ReMaxAdvantage, dict(reward_baselines=[2.0, 4.0])),
    ])
    def test_list_equals_tensor(self, estimator, kwargs):
        values = [2.0, 4.0, 6.0, 8.0]
        if estimator is GRPOPassKAdvantage:
            values = [1.0, 3.0, 2.0]
        if estimator is ReMaxAdvantage:
            values = [3.0, 5.0]
            kwargs['reward_baselines'] = [2.0, 4.0]
        from_list = estimator()(values, **kwargs)
        from_tensor = estimator()(torch.tensor(values, dtype=torch.float32), **kwargs)
        torch.testing.assert_close(from_list, from_tensor, atol=0, rtol=0)


class TestTeacherSignal:
    """T2-18: the three OPD-RL/RLSD shaping functions, hand-computed."""

    def test_compute_teacher_logratio(self):
        teacher = torch.tensor([[1.0, 2.0, 3.0]])
        policy = torch.tensor([[0.5, 2.0, 1.0]])
        mask = torch.tensor([[True, True, False]])
        out = compute_teacher_logratio(teacher, policy, mask)
        torch.testing.assert_close(out, torch.tensor([[0.5, 0.0, 0.0]]), atol=ATOL, rtol=0)

    def test_expand_advantage_broadcast_only(self):
        # Without a teacher this is a plain broadcast over the full token frame (the docstring
        # contract): masking outside the response is the loss's job, not this function's.
        adv = torch.tensor([2.0, -1.0])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        out = expand_advantage_to_per_token(adv, mask)
        torch.testing.assert_close(out, torch.tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]), atol=ATOL, rtol=0)

    def test_expand_advantage_with_teacher_signal(self):
        adv = torch.tensor([1.0])
        mask = torch.tensor([[True, True]])
        teacher = torch.tensor([[1.0, 2.0]])
        policy = torch.tensor([[0.5, 2.0]])
        out = expand_advantage_to_per_token(adv, mask, teacher, policy, teacher_kl_coef=2.0)
        # signed = [0.5, 0.0] -> adv + 2 * signed = [2.0, 1.0]
        torch.testing.assert_close(out, torch.tensor([[2.0, 1.0]]), atol=ATOL, rtol=0)

    def test_apply_rlsd_reweight_hand_computed(self):
        # One sequence, A=2 (positive): delta=[0.1, -0.2] -> w=[e^.1, e^-.2];
        # clip keeps both (range 0.9..1.1 clips e^-.2=0.819 up to 0.9); lam=1, negative_only=False.
        base = torch.tensor([2.0])
        mask = torch.tensor([[True, True]])
        teacher = torch.tensor([[0.6, 1.8]])
        policy = torch.tensor([[0.5, 2.0]])
        out = apply_rlsd_reweight(base, mask, teacher, policy, lam=1.0, clip_range=0.1)
        w0 = min(math.exp(0.1), 1.1)
        w1 = max(math.exp(-0.2), 0.9)
        torch.testing.assert_close(out, torch.tensor([[2.0 * w0, 2.0 * w1]]), atol=1e-5, rtol=0)

    def test_apply_rlsd_lam_zero_is_plain_broadcast(self):
        base = torch.tensor([2.0, -1.0])
        mask = torch.tensor([[True, True], [True, True]])
        teacher = torch.ones(2, 2)
        policy = torch.zeros(2, 2)
        out = apply_rlsd_reweight(base, mask, teacher, policy, lam=0.0, clip_range=0.1)
        torch.testing.assert_close(out, base.unsqueeze(1).expand(2, 2), atol=ATOL, rtol=0)

    def test_apply_rlsd_negative_only_keeps_positive_sequence(self):
        base = torch.tensor([2.0, -1.0])
        mask = torch.ones(2, 2)
        teacher = torch.full((2, 2), 0.6)
        policy = torch.zeros(2, 2)  # delta=0.6 everywhere -> weights clip to 1.1 at lam=1
        out = apply_rlsd_reweight(base, mask, teacher, policy, lam=1.0, clip_range=0.1, negative_only=True)
        # The positive sequence keeps its raw advantage; the negative one has sign(A)=-1, so
        # w = exp(-0.6) ~ 0.55 clips up to the lower bound 0.9 -> A * 0.9.
        torch.testing.assert_close(out[0], torch.full((2,), 2.0), atol=ATOL, rtol=0)
        torch.testing.assert_close(out[1], torch.full((2,), -0.9), atol=ATOL, rtol=0)
