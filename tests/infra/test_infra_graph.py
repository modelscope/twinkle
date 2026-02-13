import numpy as np
import unittest
from typing import List

from twinkle import DeviceGroup, DeviceMesh
from twinkle.infra import get_device_placement


class TestInfraGraph(unittest.TestCase):

    def test_print_graph(self):
        _device_group: List[DeviceGroup] = [
            DeviceGroup(
                name='training_cluster',
                ranks=list(range(16)),
                device_type='CUDAAccelerator',
                _device_mesh={
                    'main':
                    DeviceMesh(
                        device_type='cuda',
                        mesh=np.arange(16).reshape(2, 2, 4),  # pp=2, dp=2, tp=4
                        mesh_dim_names=('pp', 'dp', 'tp'),
                    ),
                }),
            DeviceGroup(
                name='inference_cluster',
                ranks=list(range(8)),
                device_type='CUDAAccelerator',
                _device_mesh={
                    'inference':
                    DeviceMesh(
                        device_type='cuda',
                        mesh=np.arange(8).reshape(2, 4),  # dp=2, tp=4
                        mesh_dim_names=('dp', 'tp'),
                    ),
                    'expert':
                    DeviceMesh(
                        device_type='cuda',
                        mesh=np.arange(8).reshape(4, 2),  # ep=4, tp=2
                        mesh_dim_names=('ep', 'tp'),
                    ),
                }),
        ]

        print(get_device_placement(_device_group))
