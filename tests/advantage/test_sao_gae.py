import pytest
import torch

from twinkle.advantage import SAOGAEAdvantage


def test_skip_observation_gae_crosses_observation():
    gae = SAOGAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, returns = gae(
        [[0.0, 99.0, 1.0]], [[0.2, 42.0, 0.4]],
        action_masks=[[True, False, True]], terminated=[True], truncated=[False])
    torch.testing.assert_close(advantages, torch.tensor([[0.8, 0.0, 0.6]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 0.0, 1.0]]))


def test_terminal_has_zero_bootstrap():
    gae = SAOGAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, _ = gae([[2.0]], [[0.5]], action_masks=[[True]], terminated=[True], truncated=[False])
    assert advantages.item() == pytest.approx(1.5)


def test_truncated_requires_and_uses_bootstrap():
    gae = SAOGAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    with pytest.raises(ValueError, match='requires a bootstrap'):
        gae([[0.0]], [[0.5]], action_masks=[[True]], terminated=[False], truncated=[True])
    advantages, _ = gae(
        [[0.0]], [[0.5]], action_masks=[[True]], terminated=[False], truncated=[True], bootstrap_values=[2.0])
    assert advantages.item() == pytest.approx(1.5)


def test_batch_sequences_do_not_link():
    gae = SAOGAEAdvantage(gamma=1.0, gae_lambda=1.0, normalize=False)
    advantages, _ = gae(
        [[1.0, 0.0], [3.0, 0.0]], torch.zeros(2, 2),
        action_masks=[[True, False], [True, False]], terminated=[True, True], truncated=[False, False])
    torch.testing.assert_close(advantages[:, 0], torch.tensor([1.0, 3.0]))


def test_length_adaptive_lambda_is_per_sequence():
    gae = SAOGAEAdvantage(gamma=1.0, alpha=1.0, normalize=False)
    advantages, _ = gae(
        [[0.0, 1.0], [0.0, 1.0]], torch.zeros(2, 2),
        action_masks=[[True, True], [True, True]], terminated=[True, True], truncated=[False, False],
        effective_lengths=[2, 4])
    assert advantages[0, 0].item() == pytest.approx(0.5)
    assert advantages[1, 0].item() == pytest.approx(0.75)
