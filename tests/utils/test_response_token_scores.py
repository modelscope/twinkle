# Copyright (c) ModelScope Contributors. All rights reserved.
"""Response positions must survive variable-length Megatron microbatches."""
import pytest
import torch

from twinkle.signals.response import response_token_scores


@pytest.mark.parametrize('serialized', [False, True])
def test_variable_length_microbatches_keep_feature_order(serialized):
    batches = [torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6., 7.]])]
    values = [batch.tolist() for batch in batches] if serialized else batches
    features = [{'labels': [-100, 1]}, {'labels': [1, -100]}, {'labels': [-100, 1, 1]}]
    assert response_token_scores({'logps': values}, features) == [[2.], [3.], [6., 7.]]


@pytest.mark.parametrize('values', [torch.tensor([[1., 2.], [3., 4.]]), [[1., 2.], [3., 4.]]])
def test_dense_rows_remain_aligned(values):
    assert response_token_scores({'logps': values}, [{'labels': [-100, 1]}] * 2) == [[2.], [4.]]


def test_microbatch_row_count_mismatch_fails():
    with pytest.raises(RuntimeError, match='per-row alignment'):
        response_token_scores({'logps': [[[1., 2.], [3., 4.]]]}, [{'labels': [-100, 1]}])
