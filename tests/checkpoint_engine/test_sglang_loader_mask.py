# Copyright (c) ModelScope Contributors. All rights reserved.
"""_masked_copy must mask the sparse-delta NaN sentinel on every write path.

defect71: sglang's mamba_v2_sharded_weight_loader writes via Tensor.__setitem__
(param.data[boundary:boundary+take] = loaded[...]), not copy_. Masking only copy_ let the NaN
sentinel leak into the model (delta_sharded loss=nan, full-sync finite). Both write paths must
mask NaN positions while leaving dense values and non-float writes untouched.
"""
import pytest
import torch

from twinkle.checkpoint_engine.delta.rollout.sglang_loader import _masked_copy


def _sentinel():
    # _densify sparse output: changed positions finite, unchanged positions NaN.
    flat = torch.full((8,), float('nan'), dtype=torch.float32)
    flat.index_copy_(0, torch.tensor([1, 4, 6]), torch.tensor([10.0, 20.0, 30.0]))
    return flat


def test_copy_masks_nan_sentinel():
    param = torch.arange(1.0, 9.0)
    with _masked_copy():
        param.copy_(_sentinel())
    assert not torch.isnan(param).any()
    assert torch.equal(param, torch.tensor([1.0, 10.0, 3.0, 4.0, 20.0, 6.0, 30.0, 8.0]))


def test_setitem_masks_nan_sentinel():
    # The defect71 path: index assignment, not copy_.
    param = torch.arange(1.0, 9.0)
    with _masked_copy():
        param[0:8] = _sentinel()
    assert not torch.isnan(param).any()
    assert torch.equal(param, torch.tensor([1.0, 10.0, 3.0, 4.0, 20.0, 6.0, 30.0, 8.0]))


def test_setitem_boundary_slice_masks_nan():
    # mamba loader writes a narrowed boundary slice of a 2D param.
    param = torch.arange(16.0).reshape(4, 4)
    loaded = torch.full((2, 4), float('nan'))
    loaded[0, 1] = 100.0
    loaded[1, 3] = 200.0
    with _masked_copy():
        param[1:3, :] = loaded
    assert not torch.isnan(param).any()
    assert param[1, 1] == 100.0 and param[1, 0] == 4.0  # changed updated, NaN position kept original
    assert param[2, 3] == 200.0 and param[2, 0] == 8.0
    assert param[0, 0] == 0.0 and param[3, 3] == 15.0  # outside the slice untouched


def test_dense_source_passes_through_unchanged():
    dense = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    for write in ('copy', 'setitem'):
        param = torch.arange(1.0, 9.0)
        with _masked_copy():
            if write == 'copy':
                param.copy_(dense)
            else:
                param[0:8] = dense
        assert torch.equal(param, dense), write


def test_non_float_and_scalar_writes_untouched():
    param = torch.zeros(4, dtype=torch.int64)
    with _masked_copy():
        param[0:4] = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        param[0] = 7
    assert torch.equal(param, torch.tensor([7, 2, 3, 4], dtype=torch.int64))


def test_mask_restored_after_context():
    param = torch.arange(1.0, 9.0)
    with _masked_copy():
        pass
    # Outside the context the NaN sentinel is NOT masked (plain assignment leaks NaN).
    param[0:8] = _sentinel()
    assert torch.isnan(param).any()