# Copyright (c) ModelScope Contributors. All rights reserved.
"""Directed tests for ``server/runtime.init_twinkle_runtime``.

The processor deployment layer has no test under ``tests/server/`` and reuses this
function for it, so this pins the parameter trap it introduces: ``ncpu_proc_per_node`` is
forwarded only when provided (processor), never when unset (model/sampler), and the
DeviceMesh is built the same way as before (``mesh_dim_names`` -> ``DeviceMesh(**)`` else
``DeviceMesh.from_sizes(**)``).
"""
from __future__ import annotations

from unittest import mock

from twinkle.server.runtime import init_twinkle_runtime


def test_model_sampler_path_does_not_forward_ncpu_proc_per_node():
    with mock.patch('twinkle.initialize') as init, mock.patch('twinkle.DeviceMesh') as mesh:
        mesh.from_sizes.return_value = 'MESH'
        result = init_twinkle_runtime(False, 2, device_group='DG', device_mesh_dict={'sizes': [2]})
    assert result == 'MESH'
    kwargs = init.call_args.kwargs
    assert 'ncpu_proc_per_node' not in kwargs
    assert kwargs['nproc_per_node'] == 2 and kwargs['groups'] == ['DG']
    mesh.from_sizes.assert_called_once_with(sizes=[2])


def test_processor_path_forwards_ncpu_proc_per_node_and_named_mesh():
    with mock.patch('twinkle.initialize') as init, mock.patch('twinkle.DeviceMesh') as mesh:
        mesh.return_value = 'NAMED_MESH'
        result = init_twinkle_runtime(
            False, 1, device_group='DG', device_mesh_dict={'mesh_dim_names': ['dp']}, ncpu_proc_per_node=8)
    assert result == 'NAMED_MESH'
    assert init.call_args.kwargs['ncpu_proc_per_node'] == 8
    mesh.assert_called_once_with(mesh_dim_names=['dp'])


def test_mock_backend_returns_none_and_uses_single_cpu_proc():
    with mock.patch('twinkle.initialize') as init, mock.patch('twinkle.DeviceMesh'):
        result = init_twinkle_runtime(True, 1, device_group='DG', device_mesh_dict={})
    assert result is None
    assert init.call_args.kwargs['ncpu_proc_per_node'] == 1
