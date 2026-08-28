# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import MagicMock, patch

import torch

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy


def test_single_rank_native_fsdp_places_model_on_local_device():
    device_mesh = MagicMock()
    device_mesh.mesh_dim_names = None
    model = MagicMock()
    moved_model = MagicMock()
    model.to.return_value = moved_model
    strategy = NativeFSDPStrategy(device_mesh=device_mesh, enable_ep=False)

    with patch(
            'twinkle.model.transformers.strategy.native_fsdp.Platform.get_local_device',
            return_value='cuda:0',
    ):
        wrapped_model, wrapped_optimizer = strategy.wrap_model(model)

    model.to.assert_called_once_with(torch.device('cuda:0'))
    assert wrapped_model is moved_model
    assert wrapped_optimizer is None
