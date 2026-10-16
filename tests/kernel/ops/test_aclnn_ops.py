"""Tests for the self-compiled ACLNN SAS/LI wrapper module."""
import inspect

import torch


def test_aclnn_ops_imports():
    from twinkle.kernel.ops.dsv4_sas_li.aclnn_ops import (
        LightningIndexer, SparseAttnSharedKV, npu_lightning_indexer, npu_sparse_attn_shared_kv,
    )
    assert callable(npu_sparse_attn_shared_kv)
    assert callable(npu_lightning_indexer)
    assert issubclass(SparseAttnSharedKV, torch.autograd.Function)
    assert issubclass(LightningIndexer, torch.autograd.Function)


def test_sas_convenience_signature():
    from twinkle.kernel.ops.dsv4_sas_li.aclnn_ops import npu_sparse_attn_shared_kv

    sig = inspect.signature(npu_sparse_attn_shared_kv)
    params = list(sig.parameters)
    expected = ['query', 'ori_kv', 'cmp_kv', 'cmp_sparse_indices', 'sinks',
                'softmax_scale', 'cmp_ratio']
    assert params[:7] == expected
    assert sig.parameters['ori_mask_mode'].default == 4
    assert sig.parameters['cmp_mask_mode'].default == 3
    assert sig.parameters['ori_win_left'].default == 127
    assert sig.parameters['ori_win_right'].default == 0
    assert sig.parameters['layout'].default == 'BSND'


def test_li_convenience_signature():
    from twinkle.kernel.ops.dsv4_sas_li.aclnn_ops import npu_lightning_indexer

    sig = inspect.signature(npu_lightning_indexer)
    params = list(sig.parameters)
    assert params == ['query', 'key', 'weights', 'sparse_count', 'sparse_mode', 'cmp_ratio']
    assert sig.parameters['sparse_count'].default == 2048
    assert sig.parameters['sparse_mode'].default == 3
    assert sig.parameters['cmp_ratio'].default == 1


def test_dsv4_npu_does_not_import_mindspeed():
    """The SAS/LI forward functions must not reference mindspeed at module level."""
    import twinkle.kernel.ops.dsv4_sas_li.npu as att_mod

    source = inspect.getsource(att_mod)
    assert 'import mindspeed' not in source
    assert 'mindspeed.ops' not in source


def test_builder_import():
    from twinkle.kernel.ops.dsv4_sas_li.aclnn.builder import build_op

    assert callable(build_op)
