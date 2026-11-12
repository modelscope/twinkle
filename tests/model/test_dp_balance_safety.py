"""DP reordering must preserve the objective and cached forward/loss alignment."""

from inspect import unwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from twinkle.loss import (
    BNPOLoss,
    ChannelLoss,
    ChunkedCrossEntropyLoss,
    CISPOLoss,
    ClipCovLoss,
    CPOLoss,
    CrossEntropyLoss,
    DPOLoss,
    GKDLoss,
    GRPOLoss,
    KLCovLoss,
    LigerFusedLinearCrossEntropyLoss,
    Loss,
    MOPDLoss,
    OPSDLoss,
    ORPOLoss,
    PPOLoss,
    PPOValueLoss,
    REALLoss,
    RewardLoss,
    SimPOLoss,
)
from twinkle.loss.base import dp_balance_group_size
from twinkle.model.optimizer_group import BaseOptimizerGroup, TrainStatus
from twinkle.model.transformers.transformers import TransformersModel
from .test_dp_balance import _call_inputs, _FakeModel
from .test_grpo_micro_batch import TrainingHarness


@pytest.mark.parametrize(
    "loss",
    [
        Loss(),
        CrossEntropyLoss(),
        PPOLoss(),
        PPOValueLoss(),
        BNPOLoss(),
        GKDLoss(),
        OPSDLoss(),
        DPOLoss(sft_weight=1.0),
        CPOLoss(),
        ORPOLoss(),
        REALLoss(beta=0.1),
        GRPOLoss(top_entropy_quantile=0.5),
        GRPOLoss(loss_type="fipo"),
        ChannelLoss(),
        ChunkedCrossEntropyLoss(chunk_size=4),
        LigerFusedLinearCrossEntropyLoss(reduction="mean"),
        CISPOLoss(),
        ClipCovLoss(),
        KLCovLoss(),
        MOPDLoss(),
    ],
)
def test_non_invariant_losses_skip_dp_reordering(loss):
    model = _FakeModel(group_size=dp_balance_group_size(loss))
    assert model._dp_balance_call("forward_backward", (), {"inputs": _call_inputs([9, 8, 1, 1])}) is None


@pytest.mark.parametrize("loss", [CrossEntropyLoss(), PPOLoss(), BNPOLoss()])
def test_rank_local_token_means_change_gradients_if_reordered(loss):
    from twinkle.model.dp_balance import plan_dp_balance_for_lengths

    lengths = [9, 8, 1, 1]
    plan = plan_dp_balance_for_lengths(lengths, 2, padding_free=True)

    def gradient(order):
        weight = torch.tensor(1.0, requires_grad=True)
        terms = []
        for start in (0, 2):
            indices = order[start : start + 2]
            values = torch.stack([weight * (index + 1) * torch.ones(9) for index in indices])
            mask = torch.stack([torch.arange(9) < lengths[index] for index in indices]).float()
            terms.append(
                loss._reduce(values, mask)["loss"]
                if isinstance(loss, CrossEntropyLoss)
                else loss._aggregate_loss(values, mask)
            )
        (sum(terms) / 2).backward()
        return weight.grad

    assert not torch.isclose(gradient(list(range(4))), gradient(plan.order))
    assert dp_balance_group_size(loss) is None


@pytest.mark.parametrize(
    "loss, size",
    [
        (GRPOLoss(), 1),
        (PPOLoss(loss_agg_mode="seq-mean-token-mean"), 1),
        (BNPOLoss(token_mean_scope="global"), 1),
        (CrossEntropyLoss(reduction="sum"), 1),
        (ChannelLoss(reduction="sum"), 1),
        (ChunkedCrossEntropyLoss(chunk_size=4, reduction="sum"), 1),
        (LigerFusedLinearCrossEntropyLoss(reduction="sum"), 1),
        (DPOLoss(), 2),
        (SimPOLoss(), 2),
        (RewardLoss(), 2),
        (REALLoss(num_generations=4), 4),
    ],
)
def test_audited_invariant_losses_keep_balancing_enabled(loss, size):
    assert dp_balance_group_size(loss) == size


@pytest.mark.parametrize(
    "factory",
    [
        GRPOLoss,
        lambda: PPOLoss(loss_agg_mode="seq-mean-token-mean"),
        lambda: BNPOLoss(token_mean_scope="global"),
        lambda: CrossEntropyLoss(reduction="sum"),
    ],
)
def test_enabled_reordering_preserves_aggregated_training_gradient(factory):
    lengths = [9, 8, 1, 1]
    rows = [
        {"input_ids": [index] * length, "labels": [1] * length, "coefficients": [0.1 * (index + 1)] * length}
        for index, length in enumerate(lengths)
    ]
    kwargs = {"inputs": rows, "advantages": [1.0, -0.5, 0.7, -1.2], "old_logps": [[0.0] * length for length in lengths]}
    planner = _FakeModel(group_size=dp_balance_group_size(factory()))
    plan, _, balanced = planner._dp_balance_call("forward_backward", (), kwargs)
    assert plan is not None and plan.reorders

    def gradient(batch):
        models = [TrainingHarness(factory()), TrainingHarness(factory())]
        for rank, model in enumerate(models):
            lo, hi = rank * 2, (rank + 1) * 2
            model.run(
                batch["inputs"][lo:hi], advantages=batch["advantages"][lo:hi], old_logps=batch["old_logps"][lo:hi]
            )
        return sum(model.weight.grad for model in models) / sum(
            model.optimizer_group["default"].train_status.num_tokens for model in models
        )

    torch.testing.assert_close(gradient(kwargs), gradient(balanced))


def _cached_model():
    group = BaseOptimizerGroup(loss_instance=GRPOLoss(), eval_status=TrainStatus())
    model = SimpleNamespace(
        optimizer_group={"default": group},
        _get_default_group=lambda: "default",
        model=SimpleNamespace(training=False),
        device_mesh=None,
    )
    model._prepare_forward_only_inputs = lambda inputs, optimizer_config: inputs

    def forward(*, inputs, optimizer_config, **kwargs):
        status = optimizer_config.eval_status
        status.inputs = {"labels": torch.ones(len(inputs), 2, dtype=torch.long)}
        status.outputs = {"logps": torch.zeros(len(inputs), 2, requires_grad=True)}
        status.loss_value = 0.0
        status.forward_kwargs = kwargs
        return dict(status.outputs)

    model._forward_only_single = forward
    return model, group


def _forward(model, **kwargs):
    return unwrap(TransformersModel.forward_only)(
        model, inputs=_call_inputs([9, 1]), _dp_balance={"reordered": True, "metrics": {}}, **kwargs
    )


@pytest.mark.parametrize("advantages", [[1.0, 2.0], torch.tensor([1.0, 2.0])])
def test_calculate_loss_uses_signals_supplied_to_reordered_forward_only(advantages):
    model, _ = _cached_model()
    _forward(model, advantages=advantages)
    assert unwrap(TransformersModel.calculate_loss)(model) == pytest.approx(-1.5)


@pytest.mark.parametrize(
    "extra",
    [
        {"advantages": [1.0, 2.0]},
        {"advantages": torch.tensor([1.0, 2.0])},
        {"advantages": np.array([1.0, 2.0])},
        {"extra": {"advantages": [1.0, 2.0]}},
        {"extra": {"advantages": torch.tensor([1.0, 2.0])}},
    ],
)
def test_calculate_loss_rejects_fresh_per_sample_arguments_after_reordering(extra):
    model, _ = _cached_model()
    _forward(model)
    with pytest.raises(ValueError, match="per-sample"):
        unwrap(TransformersModel.calculate_loss)(model, **extra)


def test_reordered_micro_batch_forward_only_cannot_calculate_loss_from_last_micro_batch():
    model, _ = _cached_model()
    model._forward_only_micro_batches = lambda **kwargs: model._forward_only_single(
        **{key: value for key, value in kwargs.items() if key != "config"}
    )
    _forward(model, advantages=[1.0, 2.0], micro_batch_size=1)
    with pytest.raises(ValueError, match="micro-batch"):
        unwrap(TransformersModel.calculate_loss)(model)


def test_reordered_forward_only_does_not_allow_switching_loss_groups():
    model, group = _cached_model()
    group.set_loss(GRPOLoss(), loss_key="policy")
    _forward(model, loss_key="policy", advantages=[1.0, 2.0])
    with pytest.raises(ValueError, match="same loss_key"):
        unwrap(TransformersModel.calculate_loss)(model)
    assert unwrap(TransformersModel.calculate_loss)(model, loss_key="policy") == pytest.approx(-1.5)


def test_unbalanced_forward_clears_cached_balanced_signals():
    model, group = _cached_model()
    _forward(model, advantages=[1.0, 2.0])
    unwrap(TransformersModel.forward_only)(model, inputs=_call_inputs([9, 1]))
    assert not group.dp_balance_reordered
    assert group.dp_balance_forward_kwargs is None
    assert unwrap(TransformersModel.calculate_loss)(model, advantages=torch.tensor([3.0, 4.0])) == pytest.approx(-3.5)
