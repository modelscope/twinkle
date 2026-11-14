# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from twinkle import remote_function
from twinkle.infra.collectors import tensor_dict
from twinkle.model.base import TrainableModel

from twinkle.model.dp_balance import (
    DPBalancePlan,
    _estimate_ranks,
    estimate_workload,
    micro_batch_workload,
    plan_dp_balance,
    plan_dp_balance_for_lengths,
)
from twinkle.model.micro_batch import MicroBatchConfig, _kk_partition


def _rank_chunks(values, dp_size):
    base, remainder = divmod(len(values), dp_size)
    chunks, start = [], 0
    for rank in range(dp_size):
        size = base + int(rank < remainder)
        chunks.append(values[start : start + size])
        start += size
    return chunks


def _peak(lengths, order, dp_size, config, padding_free):
    rank_lengths = _rank_chunks([lengths[index] for index in order], dp_size)
    return max(_estimate_ranks(rank_lengths, config, padding_free, estimate_workload).workloads)


def test_equal_size_kk_partition_keeps_group_sizes():
    groups = _kk_partition([9, 1, 8, 2, 7, 3], 2, equal_size=True)

    assert sorted(len(group) for group in groups) == [3, 3]
    assert sorted(index for group in groups for index in group) == list(range(6))
    with pytest.raises(ValueError, match="equal-size"):
        _kk_partition([1, 2, 3], 2, equal_size=True)


def test_padded_micro_batch_costs_its_longest_row():
    assert micro_batch_workload([4, 10], padding_free=False, workload_fn=lambda n: n) == 20
    assert micro_batch_workload([4, 10], padding_free=True, workload_fn=lambda n: n) == 14


@pytest.mark.parametrize(
    "padding_free, config",
    [
        (True, None),
        (False, MicroBatchConfig(micro_batch_size=2)),
        (True, MicroBatchConfig(micro_batch_size=2)),
        (False, MicroBatchConfig(micro_batch_size=4, dynamic_batching=True, max_tokens_per_micro_batch=4096)),
        (True, MicroBatchConfig(micro_batch_size=4, dynamic_batching=True, max_tokens_per_micro_batch=4096)),
    ],
)
def test_plan_keeps_rank_sizes_and_lowers_the_slowest_rank(padding_free, config):
    lengths = [2000, 1900, 1800, 1700, 300, 250, 200, 150]

    plan = plan_dp_balance_for_lengths(lengths, 2, padding_free=padding_free, micro_batch_config=config)

    assert sorted(plan.order) == list(range(len(lengths)))
    assert [len(chunk) for chunk in _rank_chunks(plan.order, 2)] == [4, 4]
    assert plan.strategy != "identity"
    assert _peak(lengths, plan.order, 2, config, padding_free) < _peak(lengths, range(8), 2, config, padding_free)
    assert plan.metrics["dp_balance/workload_imbalance_after"] < plan.metrics["dp_balance/workload_imbalance_before"]
    assert ("dp_balance/padding_ratio_after" in plan.metrics) == (not padding_free)


def test_padded_single_batch_per_rank_cannot_beat_its_longest_sequence():
    # Every rank pads its whole chunk to its longest row, so the rank holding 2000 costs 4 * w(2000) anyway.
    plan = plan_dp_balance_for_lengths([2000, 1900, 1800, 1700, 300, 250, 200, 150], 2, padding_free=False)

    assert plan.strategy == "identity"
    assert plan.order == list(range(8))


def test_padded_dynamic_batching_prefers_ranks_with_similar_lengths():
    # Real-token balance can pair a long sequence with short ones, which padded micro-batches pay for.
    lengths = [1000, 1000, 520, 480, 500, 500, 10, 10]
    config = MicroBatchConfig(micro_batch_size=4, dynamic_batching=True, max_tokens_per_micro_batch=2000)

    plan = plan_dp_balance_for_lengths(lengths, 2, padding_free=False, micro_batch_config=config)

    peaks = {
        name: _peak(lengths, order, 2, config, False)
        for name, order in {"chosen": plan.order, "identity": list(range(8))}.items()
    }
    assert peaks["chosen"] < peaks["identity"]
    assert plan.metrics["dp_balance/padding_ratio_after"] <= plan.metrics["dp_balance/padding_ratio_before"]


def test_dynamic_batching_estimate_uses_the_model_common_micro_batch_count():
    # On its own rank 1 fits in one padded micro-batch costing 3 * 30; split into rank 0's three, it costs 50.
    config = MicroBatchConfig(micro_batch_size=4, dynamic_batching=True, max_tokens_per_micro_batch=100)
    rank_lengths = [[90, 80, 70], [30, 10, 10]]

    estimate = _estimate_ranks(rank_lengths, config, False, lambda n: n)

    assert estimate.micro_batches == 3
    assert estimate.workloads == [240, 50]


def test_balanced_batch_keeps_its_order():
    plan = plan_dp_balance_for_lengths([5, 1, 1, 5], 2, padding_free=True)

    assert plan.order == [0, 1, 2, 3]
    assert plan.strategy == "identity"


def test_groups_stay_on_one_rank():
    lengths = [900, 900, 100, 100, 800, 800, 200, 200]

    plan = plan_dp_balance_for_lengths(lengths, 2, padding_free=True, group_size=2)

    pairs = [plan.order[index : index + 2] for index in range(0, 8, 2)]
    assert all(first % 2 == 0 and second == first + 1 for first, second in pairs)


def test_uneven_batch_keeps_its_order():
    # get_slice gives rank 0 three samples and rank 1 two, so a sample's loss weight depends on its rank.
    plan = plan_dp_balance_for_lengths([900, 900, 900, 100, 100], 2, padding_free=True)

    assert plan.order == [0, 1, 2, 3, 4]
    assert plan.strategy == "identity"
    assert plan.metrics == {}


def test_uneven_grouped_batch_keeps_its_order():
    plan = plan_dp_balance_for_lengths([900, 900, 100, 100] * 2 + [500, 500], 4, padding_free=True, group_size=2)

    assert plan.order == list(range(10))
    assert plan.metrics == {}


def test_unplannable_batch_is_left_for_the_model_to_reject():
    config = MicroBatchConfig(micro_batch_size=2, dynamic_batching=True, max_tokens_per_micro_batch=50)

    plan = plan_dp_balance_for_lengths([100, 10, 10, 10], 2, padding_free=True, micro_batch_config=config)

    assert plan.order == [0, 1, 2, 3]
    assert plan.metrics == {}


def test_plan_reads_encoded_inputs_and_reorders_aligned_fields():
    inputs = [{"input_ids": torch.ones(length, dtype=torch.long)} for length in (800, 700, 50, 40)]

    plan = plan_dp_balance(inputs, 2, padding_free=True)

    assert isinstance(plan, DPBalancePlan)
    assert plan.apply([0, 1, 2, 3]) == plan.order
    assert plan.apply(tuple("abcd")) == tuple("abcd"[index] for index in plan.order)
    assert plan.apply(torch.arange(4)).tolist() == plan.order
    with pytest.raises(ValueError, match="cannot reorder"):
        plan.apply([1, 2])


def test_reorder_call_moves_every_batch_aligned_argument_together():
    plan = DPBalancePlan(order=[2, 0, 3, 1], strategy="kk", metrics={})
    inputs = [{"input_ids": [index] * (index + 1)} for index in range(4)]
    old_logps = [[float(index)] for index in range(4)]

    args, kwargs = plan.reorder_call(
        (inputs,),
        {
            "old_logps": old_logps,
            "advantages": torch.arange(4.0),
            "extra": {"ref_logps": torch.arange(8).view(4, 2)},
            "micro_batch_size": 2,
            "short": [1, 2],
        },
    )

    assert args[0] == [inputs[index] for index in plan.order]
    assert kwargs["old_logps"] == [old_logps[index] for index in plan.order]
    assert kwargs["advantages"].tolist() == [2.0, 0.0, 3.0, 1.0]
    assert kwargs["extra"]["ref_logps"][:, 0].tolist() == [4, 0, 6, 2]
    assert kwargs["micro_batch_size"] == 2
    assert kwargs["short"] == [1, 2]


def test_restore_outputs_undoes_the_reorder():
    plan = DPBalancePlan(order=[2, 0, 3, 1], strategy="kk", metrics={})
    outputs = {"logps": torch.arange(12).view(4, 3), "loss": 1.5, "sequences": list("abcd")}

    _, shuffled = plan.reorder_call((), {"outputs": outputs})
    restored = plan.restore_outputs(shuffled["outputs"])

    assert torch.equal(restored["logps"], outputs["logps"])
    assert restored["sequences"] == outputs["sequences"]
    assert restored["loss"] == 1.5


class _FakeModel:
    _dp_balance_call = TrainableModel._dp_balance_call

    def __init__(self, enabled=True, group_size=1, data_world_size=2):
        self.device_mesh = SimpleNamespace(data_world_size=data_world_size)
        self.spec = {"enabled": enabled, "padding_free": True, "hidden_size": 1024, "group_size": group_size}
        self.spec_calls = 0

    def _dp_balance_spec(self, adapter_name=None, loss_key=None):
        self.spec_calls += 1
        return self.spec


def _call_inputs(lengths):
    return [{"input_ids": torch.ones(length, dtype=torch.long)} for length in lengths]


def test_model_reorders_the_call_and_tells_the_workers():
    inputs = _call_inputs([900, 800, 700, 600, 50, 40, 30, 20])
    advantages = list(range(8))

    plan, args, kwargs = _FakeModel()._dp_balance_call(
        "forward_backward", (), {"inputs": inputs, "advantages": advantages}
    )

    assert plan is not None and plan.reorders
    assert [len(item["input_ids"]) for item in kwargs["inputs"]] == [len(inputs[i]["input_ids"]) for i in plan.order]
    assert kwargs["advantages"] == plan.order
    assert kwargs["_dp_balance"]["reordered"] is True
    assert "dp_balance/workload_imbalance_after" in kwargs["_dp_balance"]["metrics"]


def test_model_keeps_preference_pairs_together():
    inputs = _call_inputs([900, 900, 800, 800, 50, 50, 30, 30])

    plan, _, _ = _FakeModel(group_size=2)._dp_balance_call("forward_backward", (), {"inputs": inputs})

    pairs = [plan.order[index : index + 2] for index in range(0, 8, 2)]
    assert all(first % 2 == 0 and second == first + 1 for first, second in pairs)


@pytest.mark.parametrize(
    "model, extra",
    [
        (_FakeModel(enabled=False), {}),
        (_FakeModel(group_size=None), {}),
        (_FakeModel(data_world_size=1), {}),
        (_FakeModel(), {"return_logits": True}),
        (_FakeModel(), {"group_scores": [0.5, 0.25]}),
    ],
)
def test_model_leaves_the_call_alone(model, extra):
    kwargs = {"inputs": _call_inputs([900, 800, 700, 600, 50, 40, 30, 20]), **extra}

    assert model._dp_balance_call("forward_backward", (), kwargs) is None


def test_disabled_model_is_asked_once():
    model = _FakeModel(enabled=False)
    kwargs = {"inputs": _call_inputs([900, 800, 700, 600, 50, 40, 30, 20])}

    assert model._dp_balance_call("forward_backward", (), kwargs) is None
    assert model._dp_balance_call("forward_backward", (), kwargs) is None
    assert model.spec_calls == 1


def test_group_size_follows_the_loss():
    from twinkle.loss import CrossEntropyLoss, DPOLoss
    from twinkle.loss.base import dp_balance_group_size

    assert dp_balance_group_size(None) == 1
    assert dp_balance_group_size(CrossEntropyLoss()) is None
    assert dp_balance_group_size(CrossEntropyLoss(reduction='sum')) == 1
    assert dp_balance_group_size(DPOLoss()) == 2
    assert dp_balance_group_size(lambda inputs, outputs: 0) is None


class _LocalModel:
    _dp_balance_local = TrainableModel._dp_balance_local

    def __init__(self, mesh):
        self.device_mesh = mesh
        self.seen = None

    def _dp_balance_spec(self, adapter_name=None, loss_key=None):
        return {"enabled": True, "padding_free": True, "hidden_size": 1024, "group_size": 1}

    @remote_function(dispatch="slice_dp", collect=tensor_dict, dp_balance=True)
    def forward_backward(self, *, inputs, advantages, _dp_balance=None):
        lengths = [len(model_input["input_ids"]) for model_input in inputs]
        self.seen = (lengths, advantages.tolist(), _dp_balance)
        return {
            "loss": torch.tensor(float(sum(lengths))),
            "lengths": torch.tensor(lengths),
            "logps": [torch.full((length,), float(length)) for length in lengths],
        }


def _local_worker(rank, rendezvous, result_dir):
    import os

    import torch.distributed as dist

    from twinkle import infra
    from twinkle.utils import DeviceMesh

    os.environ.update(RANK=str(rank), WORLD_SIZE="2")
    infra._mode = "local"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        lengths = [900, 800, 700, 600, 50, 40, 30, 20][rank * 4 : (rank + 1) * 4]
        model = _LocalModel(DeviceMesh.from_sizes(world_size=2, dp_size=2, device_type="cpu"))
        output = model.forward_backward(
            inputs=_call_inputs(lengths), advantages=torch.tensor(lengths, dtype=torch.float)
        )
        seen_lengths, seen_advantages, balance = model.seen
        torch.save(
            {
                "seen": seen_lengths,
                "advantages_aligned": seen_advantages == [float(length) for length in seen_lengths],
                "reordered": balance["reordered"],
                "restored": output["lengths"].tolist() == lengths
                and [len(logps) for logps in output["logps"]] == lengths,
                "own_loss": float(output["loss"]) == float(sum(seen_lengths)),
            },
            f"{result_dir}/{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_local_mode_swaps_samples_and_returns_outputs_to_their_rank(tmp_path):
    torch.multiprocessing.spawn(_local_worker, args=(str(tmp_path / "gloo"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"{rank}.pt") for rank in range(2)]

    assert all(result["reordered"] and result["advantages_aligned"] for result in results)
    assert all(result["restored"] and result["own_loss"] for result in results)
    assert sorted(results[0]["seen"] + results[1]["seen"]) == [20, 30, 40, 50, 600, 700, 800, 900]
    assert max(sum(result["seen"]) for result in results) < 900 + 800 + 700 + 600
