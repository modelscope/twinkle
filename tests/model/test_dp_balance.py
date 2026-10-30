# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

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
