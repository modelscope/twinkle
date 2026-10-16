import pytest
import torch

from twinkle.advantage import GAEAdvantage


def test_single_terminal_token():
    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)([[2.0]], [[0.5]])
    torch.testing.assert_close(advantages, torch.tensor([[1.5]]))
    torch.testing.assert_close(returns, torch.tensor([[2.0]]))


def test_multi_token_terminal_gae():
    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)(
        [[0.0, 0.0, 1.0]], [[0.2, 0.3, 0.4]])
    torch.testing.assert_close(advantages, torch.tensor([[0.8, 0.7, 0.6]]))
    torch.testing.assert_close(returns, torch.ones(1, 3))


def test_padding_is_ignored():
    advantages, returns = GAEAdvantage(normalize=False)(
        [[0.0, 1.0, 99.0]], [[0.2, 0.3, 42.0]], masks=[[True, True, False]])
    assert advantages[0, 2] == 0
    assert returns[0, 2] == 0


def test_advantage_normalization():
    advantages, _ = GAEAdvantage(gamma=0.0, gae_lambda=0.0)(
        [[1.0, 2.0], [3.0, 4.0]], torch.zeros(2, 2))
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-6)


def test_terminal_reward_and_kl_shaping():
    rewards = GAEAdvantage.build_token_rewards(
        [2.0], [2], old_logps=[[-1.0, -2.0]], ref_logps=[[-1.5, -1.5]], kl_coef=0.1)
    assert rewards[0] == pytest.approx([-0.05, 2.05])


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        GAEAdvantage(gamma=1.1)
    with pytest.raises(ValueError):
        GAEAdvantage(gae_lambda=-0.1)
