# Copyright (c) Alibaba, Inc. and its affiliates.
"""defect78-s1 permanent gate: slice_dp collection must reassemble in DATA-RANK order.

PLAN (case | input | expected | failure it kills):
1 fsdp2dp2 stride1 row identity | real from_sizes(fsdp2,dp2) mesh, outputs[w] tagged dr(w)
  | dr order 0..3 | worker-order concat permutation (the exact old red: response-framed
  logps misaligned vs trajectory order)
2 fsdp2dp2 tensor_dict list channel | per-worker {'logps': [row tensors tagged dr]}
  | stacked rows in dr order | same permutation through the tensor_dict list-merge path
3 stride>1 membership (tp2 fsdp2dp2, 8 ranks / 4 actors) | 4 actor outputs tagged dr(actor)
  | all 4 shards exactly once, dr order | old membership filter compared WORKER indices
  against GLOBAL ranks -> wrong subset under stride>1 (dropped shards)
4 pure dp4 identity | from_sizes(dp4) | identity order | over-reordering regression on the
  always-green path
5 fsdp4 dp1 identity | from_sizes(fsdp4) | identity order | same
6 unequal shard row counts | lists of lengths 1,3,2,4 | exact per-shard rows once, dr order
  | offset/padding misalignment class (ref_logps 388 vs 327)
7 single output | one worker's list | passthrough unchanged | single-worker micro-batch frame
  contract
8 empty | [] | [] / {} | empty guard

Oracle: expected data rank per worker computed independently from the mesh ARRAY coordinates
(dr = dp_idx * fsdp_ws + fsdp_idx), never via _data_rank_order/get_data_rank_from_global_rank.
Reverse-verify: restoring the old membership filter (worker order) turns cases 1/2/3/6 RED,
cases 4/5/7/8 stay GREEN.
"""
import numpy as np
import pytest
import torch

from twinkle import DeviceMesh
from twinkle.infra.collectors import dp_flatten, tensor_dict


def _worker_data_ranks(mesh, n_workers):
    """Independent oracle: actor i owns global ranks [i*stride, (i+1)*stride); read its
    (fsdp, dp) coordinates straight from the mesh array and compute the dispatch data rank."""
    stride = max(1, mesh.world_size // n_workers)
    array = np.asarray(mesh.mesh)
    names = list(mesh.mesh_dim_names)
    fsdp_ws = array.shape[names.index('fsdp')] if 'fsdp' in names else 1
    coord_of = {int(array[c]): c for c in np.ndindex(array.shape)}
    ranks = []
    for i in range(n_workers):
        coord = coord_of[i * stride]
        dp = coord[names.index('dp')] if 'dp' in names else 0
        fsdp = coord[names.index('fsdp')] if 'fsdp' in names else 0
        ranks.append(dp * fsdp_ws + fsdp)
    return ranks


def _tagged_lists(mesh, n_workers, rows_per_shard=None):
    drs = _worker_data_ranks(mesh, n_workers)
    outputs = []
    for dr in drs:
        n = 1 if rows_per_shard is None else rows_per_shard[dr]
        outputs.append([f'dr{dr}_row{j}' for j in range(n)])
    return outputs, drs


def test_fsdp2dp2_row_identity():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2)
    outputs, drs = _tagged_lists(mesh, 4)
    assert drs == [0, 2, 1, 3]  # the transpose that permuted worker-order collection
    merged = dp_flatten(outputs, mesh)
    assert merged == [f'dr{dr}_row0' for dr in range(4)]


def test_fsdp2dp2_tensor_dict_list_channel():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2)
    drs = _worker_data_ranks(mesh, 4)
    outputs = [{'logps': [torch.full((1, 2), float(dr))]} for dr in drs]
    merged = tensor_dict(outputs, mesh)
    rows = merged['logps'][:, 0].tolist()
    assert rows == [0.0, 1.0, 2.0, 3.0]


def test_stride_gt1_keeps_all_nominated_actors():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2, tp_size=2)
    assert mesh.world_size == 8
    outputs, drs = _tagged_lists(mesh, 4)  # 4 actors, stride 2
    assert drs == [0, 2, 1, 3]
    merged = dp_flatten(outputs, mesh)
    assert merged == [f'dr{dr}_row0' for dr in range(4)]


@pytest.mark.parametrize('kwargs,n_workers', [
    ({'dp_size': 4}, 4),
    ({'fsdp_size': 4}, 4),
])
def test_identity_meshes_unchanged(kwargs, n_workers):
    mesh = DeviceMesh.from_sizes(**kwargs)
    outputs, drs = _tagged_lists(mesh, n_workers)
    assert drs == list(range(n_workers))
    assert dp_flatten(outputs, mesh) == [item for sublist in outputs for item in sublist]


def test_unequal_shard_row_counts():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2)
    rows_per_shard = {0: 1, 1: 3, 2: 2, 3: 4}
    outputs, _ = _tagged_lists(mesh, 4, rows_per_shard)
    merged = dp_flatten(outputs, mesh)
    expected = [f'dr{dr}_row{j}' for dr in range(4) for j in range(rows_per_shard[dr])]
    assert merged == expected


def test_single_output_passthrough():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2)
    assert dp_flatten([['a', 'b']], mesh) == ['a', 'b']
    single = {'logps': [torch.zeros(1, 2)]}
    assert tensor_dict([single], mesh)['logps'].shape == (1, 2)


def test_empty_outputs():
    mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=2)
    assert dp_flatten([], mesh) == []
    assert tensor_dict([], mesh) == {}
