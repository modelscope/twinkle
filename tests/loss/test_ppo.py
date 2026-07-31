import torch

from twinkle.advantage import GAEAdvantage
from twinkle.loss import GRPOLoss, PPOLoss, PPOValueLoss


def test_ppo_policy_loss_reuses_grpo_objective():
    labels = torch.tensor([[1, 2, -100]])
    logps = torch.tensor([[-0.8, -1.2, 0.0]], requires_grad=True)
    old_logps = [[-1.0, -1.0]]
    advantages = [[1.0, -1.0]]
    inputs = {'labels': labels}
    outputs = {'logps': logps}
    ppo = PPOLoss(epsilon=0.2)(inputs, outputs, old_logps=old_logps, advantages=advantages)['loss']
    grpo = GRPOLoss(epsilon=0.2)(inputs, outputs, old_logps=old_logps, advantages=advantages)['loss']
    torch.testing.assert_close(ppo, grpo)


def test_value_loss_without_clipping():
    values = torch.tensor([[0.0, 1.0, 10.0]], requires_grad=True)
    result = PPOValueLoss(epsilon=0.2)(
        {'labels': torch.tensor([[1, 2, -100]])},
        {'values': values},
        old_values=[[0.0, 1.0]],
        returns=[[1.0, 1.0]],
    )
    assert result['loss'].item() == 0.25
    result['loss'].backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 2] == 0


def test_value_loss_uses_clipped_maximum():
    values = torch.tensor([[2.0]], requires_grad=True)
    result = PPOValueLoss(epsilon=0.2)(
        {'labels': torch.tensor([[1]])},
        {'values': values},
        old_values=[[0.0]],
        returns=[[1.0]],
    )
    assert result['loss'].item() == 0.5


def test_value_loss_accepts_trailing_singleton():
    values = torch.tensor([[[0.5], [0.0]]], requires_grad=True)
    result = PPOValueLoss()(
        {'labels': torch.tensor([[1, -100]])},
        {'values': values},
        old_values=[0.5],
        returns=[1.0],
    )
    assert torch.isfinite(result['loss'])


def test_ppo_rollout_updates_actor_and_critic():
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, 6]])
    old_logps = [[-0.9, -1.1, -1.0], [-1.2, -0.8, -1.1]]
    old_values = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]]
    token_rewards = GAEAdvantage.build_token_rewards([1.0, -0.5], [3, 3])
    advantages, returns = GAEAdvantage(gamma=1.0, gae_lambda=0.95, normalize=True)(
        token_rewards, old_values)

    actor_logps = torch.nn.Parameter(torch.tensor([
        [0.0, -0.7, -1.3, -0.8],
        [0.0, -1.0, -0.7, -1.4],
    ]))
    critic_values = torch.nn.Parameter(torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    optimizer = torch.optim.SGD([actor_logps, critic_values], lr=0.1)
    actor_before = actor_logps.detach().clone()
    critic_before = critic_values.detach().clone()

    actor_loss = PPOLoss(epsilon=0.2)(
        {'labels': labels},
        {'logps': actor_logps},
        old_logps=old_logps,
        advantages=advantages,
    )['loss']
    critic_loss = PPOValueLoss(epsilon=0.2)(
        {'labels': labels},
        {'values': critic_values},
        old_values=old_values,
        returns=returns,
    )['loss']
    (actor_loss + critic_loss).backward()
    optimizer.step()

    assert torch.isfinite(actor_loss)
    assert torch.isfinite(critic_loss)
    assert not torch.equal(actor_logps.detach(), actor_before)
    assert not torch.equal(critic_values.detach(), critic_before)
