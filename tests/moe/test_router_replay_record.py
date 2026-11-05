# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from twinkle.model.megatron.moe.router_replay import check_replay_indices, replay_index_dtype


@pytest.mark.parametrize('num_experts,expected', [(8, torch.uint8), (256, torch.uint8), (257, torch.int32), (None,
                                                                                                            torch.int32)])
def test_replay_index_dtype_covers_every_expert(num_experts, expected):
    """uint8 only when it can represent every expert id; a wider record must widen the dtype."""
    assert replay_index_dtype(num_experts) == expected


def test_check_replay_indices_rejects_out_of_range():
    """An id at or above num_experts would silently address another expert on replay."""
    with pytest.raises(RuntimeError, match=r'outside \[0, 128\)'):
        check_replay_indices(torch.tensor([[0, 300]]), 128)


def test_check_replay_indices_accepts_valid_and_empty():
    check_replay_indices(torch.tensor([[0, 127]]), 128)
    check_replay_indices(torch.empty((0, ), dtype=torch.int32), 128)
    check_replay_indices(torch.tensor([[0, 5]]), None)
