# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import numpy as np
from unittest.mock import patch

import twinkle
from twinkle.utils.platform import DeviceMesh, Platform

twinkle.initialize(mode='local')


class TestDeviceMeshRanks:
    def test_dp_rank_only(self):
        mesh = DeviceMesh.from_sizes(dp_size=4)
        
        for rank in range(4):
            with patch.object(Platform, 'get_rank', return_value=rank):
                assert mesh.dp_rank == rank
                assert mesh.tp_rank is None
                assert mesh.pp_rank is None
                assert mesh.fsdp_rank is None

    def test_tp_rank_only(self):
        mesh = DeviceMesh.from_sizes(tp_size=4)
        # from_sizes 默认 dp_size=1，维度顺序是 (dp, tp)
        mesh_array = mesh.mesh.reshape(1, 4)
        
        for tp_idx in range(4):
            global_rank = int(mesh_array[0, tp_idx])
            with patch.object(Platform, 'get_rank', return_value=global_rank):
                assert mesh.tp_rank == tp_idx
                assert mesh.dp_rank == 0  # dp 默认是 1，所以 dp_rank 总是 0
                assert mesh.pp_rank is None
                assert mesh.fsdp_rank is None

    def test_pp_rank_only(self):
        mesh = DeviceMesh.from_sizes(pp_size=4)
        # from_sizes 维度顺序是 (pp, dp)，默认 dp_size=1
        mesh_array = mesh.mesh.reshape(4, 1)
        
        for pp_idx in range(4):
            global_rank = int(mesh_array[pp_idx, 0])
            with patch.object(Platform, 'get_rank', return_value=global_rank):
                assert mesh.pp_rank == pp_idx
                assert mesh.dp_rank == 0  # dp 默认是 1，所以 dp_rank 总是 0
                assert mesh.tp_rank is None
                assert mesh.fsdp_rank is None

    def test_fsdp_rank_only(self):
        mesh = DeviceMesh.from_sizes(fsdp_size=4)
        # from_sizes 维度顺序是 (fsdp, dp)，默认 dp_size=1
        mesh_array = mesh.mesh.reshape(4, 1)
        
        for fsdp_idx in range(4):
            global_rank = int(mesh_array[fsdp_idx, 0])
            with patch.object(Platform, 'get_rank', return_value=global_rank):
                assert mesh.fsdp_rank == fsdp_idx
                assert mesh.dp_rank == 0  # dp 默认是 1，所以 dp_rank 总是 0
                assert mesh.tp_rank is None
                assert mesh.pp_rank is None

    def test_dp_tp_combination(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, tp_size=4)
        
        mesh_array = mesh.mesh.reshape(2, 4)
        
        for dp_idx in range(2):
            for tp_idx in range(4):
                global_rank = int(mesh_array[dp_idx, tp_idx])
                with patch.object(Platform, 'get_rank', return_value=global_rank):
                    assert mesh.dp_rank == dp_idx
                    assert mesh.tp_rank == tp_idx
                    assert mesh.pp_rank is None
                    assert mesh.fsdp_rank is None

    def test_dp_fsdp_combination(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, fsdp_size=4)
        # from_sizes 维度顺序是 (fsdp, dp)
        mesh_array = mesh.mesh.reshape(4, 2)
        
        for fsdp_idx in range(4):
            for dp_idx in range(2):
                global_rank = int(mesh_array[fsdp_idx, dp_idx])
                with patch.object(Platform, 'get_rank', return_value=global_rank):
                    assert mesh.fsdp_rank == fsdp_idx
                    assert mesh.dp_rank == dp_idx
                    assert mesh.tp_rank is None
                    assert mesh.pp_rank is None

    def test_tp_pp_combination(self):
        mesh = DeviceMesh.from_sizes(tp_size=2, pp_size=4)
        # from_sizes 维度顺序是 (pp, dp, tp)，默认 dp_size=1
        mesh_array = mesh.mesh.reshape(4, 1, 2)
        
        for pp_idx in range(4):
            for tp_idx in range(2):
                global_rank = int(mesh_array[pp_idx, 0, tp_idx])
                with patch.object(Platform, 'get_rank', return_value=global_rank):
                    assert mesh.pp_rank == pp_idx
                    assert mesh.tp_rank == tp_idx
                    assert mesh.dp_rank == 0  # dp 默认是 1，所以 dp_rank 总是 0
                    assert mesh.fsdp_rank is None

    def test_dp_tp_pp_combination(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, tp_size=2, pp_size=2)
        # from_sizes 维度顺序是 (pp, dp, tp)
        mesh_array = mesh.mesh.reshape(2, 2, 2)
        
        for pp_idx in range(2):
            for dp_idx in range(2):
                for tp_idx in range(2):
                    global_rank = int(mesh_array[pp_idx, dp_idx, tp_idx])
                    with patch.object(Platform, 'get_rank', return_value=global_rank):
                        assert mesh.pp_rank == pp_idx
                        assert mesh.dp_rank == dp_idx
                        assert mesh.tp_rank == tp_idx
                        assert mesh.fsdp_rank is None

    def test_dp_fsdp_tp_combination(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, fsdp_size=2, tp_size=2)
        # from_sizes 维度顺序是 (fsdp, dp, tp)
        mesh_array = mesh.mesh.reshape(2, 2, 2)
        
        for fsdp_idx in range(2):
            for dp_idx in range(2):
                for tp_idx in range(2):
                    global_rank = int(mesh_array[fsdp_idx, dp_idx, tp_idx])
                    with patch.object(Platform, 'get_rank', return_value=global_rank):
                        assert mesh.fsdp_rank == fsdp_idx
                        assert mesh.dp_rank == dp_idx
                        assert mesh.tp_rank == tp_idx
                        assert mesh.pp_rank is None

    def test_all_dimensions_combination(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, fsdp_size=2, tp_size=2, pp_size=2)
        # from_sizes 维度顺序是 (fsdp, pp, dp, tp)
        mesh_array = mesh.mesh.reshape(2, 2, 2, 2)
        
        for fsdp_idx in range(2):
            for pp_idx in range(2):
                for dp_idx in range(2):
                    for tp_idx in range(2):
                        global_rank = int(mesh_array[fsdp_idx, pp_idx, dp_idx, tp_idx])
                        with patch.object(Platform, 'get_rank', return_value=global_rank):
                            assert mesh.fsdp_rank == fsdp_idx
                            assert mesh.pp_rank == pp_idx
                            assert mesh.dp_rank == dp_idx
                            assert mesh.tp_rank == tp_idx

    def test_custom_mesh(self):
        mesh_array = np.arange(16).reshape(2, 2, 4)
        mesh = DeviceMesh(
            mesh=mesh_array,
            mesh_dim_names=('pp', 'dp', 'tp')
        )
        
        for pp_idx in range(2):
            for dp_idx in range(2):
                for tp_idx in range(4):
                    global_rank = int(mesh_array[pp_idx, dp_idx, tp_idx])
                    with patch.object(Platform, 'get_rank', return_value=global_rank):
                        assert mesh.pp_rank == pp_idx
                        assert mesh.dp_rank == dp_idx
                        assert mesh.tp_rank == tp_idx
                        assert mesh.fsdp_rank is None

    def test_rank_not_in_mesh(self):
        mesh = DeviceMesh.from_sizes(dp_size=4)
        
        with patch.object(Platform, 'get_rank', return_value=100):
            assert mesh.dp_rank is None
            assert mesh.tp_rank is None
            assert mesh.pp_rank is None
            assert mesh.fsdp_rank is None

    def test_world_sizes(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, fsdp_size=3, tp_size=4, pp_size=5)
        
        assert mesh.dp_world_size == 2
        assert mesh.fsdp_world_size == 3
        assert mesh.tp_world_size == 4
        assert mesh.pp_world_size == 5
        assert mesh.world_size == 2 * 3 * 4 * 5

    def test_data_rank_with_dp_only(self):
        mesh = DeviceMesh.from_sizes(dp_size=4)
        
        for rank in range(4):
            with patch.object(Platform, 'get_rank', return_value=rank):
                assert mesh.data_rank == rank

    def test_data_rank_with_fsdp_only(self):
        mesh = DeviceMesh.from_sizes(fsdp_size=4)
        
        for rank in range(4):
            with patch.object(Platform, 'get_rank', return_value=rank):
                assert mesh.data_rank == rank

    def test_data_rank_with_dp_fsdp(self):
        mesh = DeviceMesh.from_sizes(dp_size=2, fsdp_size=3)
        # from_sizes 维度顺序是 (fsdp, dp)
        mesh_array = mesh.mesh.reshape(3, 2)
        
        for fsdp_idx in range(3):
            for dp_idx in range(2):
                global_rank = int(mesh_array[fsdp_idx, dp_idx])
                with patch.object(Platform, 'get_rank', return_value=global_rank):
                    # data_rank 的计算公式: dp_rank * fsdp_world_size + fsdp_rank
                    expected_data_rank = dp_idx * 3 + fsdp_idx
                    assert mesh.data_rank == expected_data_rank
