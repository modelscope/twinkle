import math

import pytest
import torch

from twinkle.loss import SAOLoss, SAOValueLoss
from twinkle.loss.policy_objective import DISPolicyObjective, PolicyObjective


def _loss(ratios, advantages=None):
    logps = torch.tensor([[math.log(value) for value in ratios]], requires_grad=True)
    result = SAOLoss(epsilon_low=0.3, epsilon_high=5.0)(
        {'labels': torch.ones_like(logps, dtype=torch.long)},
        {'logps': logps},
        old_logps=torch.zeros_like(logps),
        advantages=advantages or [[1.0] * len(ratios)],
    )['loss']
    return logps, result


def test_sao_ratio_boundaries_are_strict():
    logps, loss = _loss([0.7, 0.7001, 5.999, 6.0])
    loss.backward()
    assert logps.grad[0, 0] == 0
    assert logps.grad[0, 1] != 0
    assert logps.grad[0, 2] != 0
    assert logps.grad[0, 3] == 0


def test_sao_outside_trust_has_zero_gradient():
    logps, loss = _loss([0.1, 7.0])
    loss.backward()
    torch.testing.assert_close(logps.grad, torch.zeros_like(logps))


def test_sao_inside_gradient_uses_detached_ratio():
    logps, loss = _loss([2.0])
    loss.backward()
    assert logps.grad.item() == pytest.approx(-2.0)


def test_sao_uses_reusable_dis_policy_objective():
    loss = SAOLoss(epsilon_low=0.3, epsilon_high=5.0)
    assert isinstance(loss.policy_objective, PolicyObjective)
    assert isinstance(loss.policy_objective, DISPolicyObjective)


def test_dis_policy_objective_matches_original_sao_formula():
    logps = torch.tensor([[math.log(0.7), math.log(2.0), math.log(6.0)]], requires_grad=True)
    ratio = torch.exp(logps)
    advantages = torch.tensor([[1.0, -0.5, 1.0]])
    objective = DISPolicyObjective(epsilon_low=0.3, epsilon_high=5.0)

    actual = objective(ratio, advantages, logps)

    trusted = (ratio > 0.7) & (ratio < 6.0)
    weight = torch.where(trusted, ratio, torch.zeros_like(ratio)).detach()
    expected = -weight * advantages.detach() * logps.float()
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    torch.testing.assert_close(logps.grad, torch.tensor([[0.0, 1.0, 0.0]]))


def test_sao_ragged_alignment_and_token_mean_denominator():
    logps = torch.tensor([[0.0, math.log(7.0), 0.0], [math.log(2.0), 0.0, 0.0]], requires_grad=True)
    result = SAOLoss()(
        {'labels': torch.tensor([[1, 2, -100], [3, -100, -100]])},
        {'logps': logps},
        old_logps=[[0.0, 0.0], [0.0]],
        advantages=[[1.0, 1.0], [1.0]],
    )['loss']
    result.backward()
    # Three action tokens form the denominator; the rejected ratio=7 token contributes zero.
    assert logps.grad[0, 0].item() == pytest.approx(-1 / 3)
    assert logps.grad[0, 1].item() == 0
    assert logps.grad[1, 0].item() == pytest.approx(-2 / 3)


def test_sao_value_loss_is_masked_mse():
    values = torch.tensor([[0.0, 2.0, 99.0]], requires_grad=True)
    loss = SAOValueLoss()(
        {'labels': torch.tensor([[1, 2, -100]])}, {'values': values}, returns=[[1.0, 0.0]])['loss']
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert values.grad[0, 2] == 0
