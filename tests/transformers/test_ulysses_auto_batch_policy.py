# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np

from twinkle import DeviceMesh
from twinkle.model.transformers.transformers import _default_gradient_accumulation_steps_for_device_mesh


class TestUlyssesAutoBatchPolicy:

    def test_default_gradient_accumulation_steps_scales_with_ulysses(self):
        device_mesh = DeviceMesh(
            device_type='cpu',
            mesh=np.arange(4).reshape(2, 2),
            mesh_dim_names=('dp', 'fsdp'),
            ulysses_size=2,
        )

        assert _default_gradient_accumulation_steps_for_device_mesh(device_mesh) == 2
