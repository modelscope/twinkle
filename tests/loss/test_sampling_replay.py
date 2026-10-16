# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from twinkle.loss import GRPOLoss


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"beta": 0.1}, "KL penalty"),
        ({"entropy_coef": 0.1}, "entropy bonus"),
    ],
)
def test_sampling_replay_rejects_incompatible_grpo_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GRPOLoss(enable_sampling_replay=True, **kwargs)


def test_sampling_replay_requires_rollout_and_model_logps():
    loss = GRPOLoss(enable_sampling_replay=True)
    inputs = {"labels": torch.tensor([[1]])}

    with pytest.raises(ValueError, match="old_logps are required"):
        loss(inputs, {"logps": torch.tensor([[-0.5]])}, advantages=[1.0])
    with pytest.raises(RuntimeError, match="must be computed by the model forward"):
        loss(
            inputs,
            {"logits": torch.zeros(1, 1, 2)},
            old_logps=[[-0.5]],
            advantages=[1.0],
        )


def test_sampling_replay_grpo_uses_replayed_importance_ratio_and_clipping():
    labels = torch.tensor([[-100, 1, 2]])
    replayed_logps = torch.tensor([[0.0, -0.4, -0.6]], requires_grad=True)
    old_logps = [[-0.5, -0.5]]
    advantages = [[1.0, -2.0]]

    result = GRPOLoss(enable_sampling_replay=True, epsilon=0.2)(
        {"labels": labels},
        {"logps": replayed_logps},
        old_logps=old_logps,
        advantages=advantages,
    )

    ratio = torch.exp(torch.tensor([0.1, -0.1]))
    clipped = ratio.clamp(0.8, 1.2)
    expected_tokens = -torch.minimum(
        ratio * torch.tensor([1.0, -2.0]), clipped * torch.tensor([1.0, -2.0])
    )
    torch.testing.assert_close(result["loss"], expected_tokens.mean())
    result["loss"].backward()
    assert replayed_logps.grad[0, 0] == 0
    assert replayed_logps.grad[0, 1:].abs().sum() > 0


def test_sampling_replay_without_advantages_returns_graph_connected_zero():
    logps = torch.tensor([[-0.2, -0.3]], requires_grad=True)
    result = GRPOLoss(enable_sampling_replay=True)(
        {"labels": torch.tensor([[1, 2]])},
        {"logps": logps},
        old_logps=[[-0.2, -0.3]],
    )

    assert result["loss"].item() == 0.0
    assert result["num_tokens"] == 0
    result["loss"].backward()
    assert torch.equal(logps.grad, torch.zeros_like(logps))
