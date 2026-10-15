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


def test_single_rank_optimizer_resume_without_process_group(tmp_path):
    from copy import deepcopy

    from twinkle import DeviceMesh

    assert not torch.distributed.is_initialized()
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    inputs = torch.randn(4, 3)

    def step(network, optim):
        optim.zero_grad()
        network(inputs).square().mean().backward()
        optim.step()

    step(model, optimizer)
    restored = deepcopy(model)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.2, foreach=False)
    strategy = NativeFSDPStrategy(device_mesh=DeviceMesh.from_sizes(fsdp_size=1, device_type='cpu'))
    checkpoint = str(tmp_path / 'optimizer.pt')
    strategy.save_optimizer_checkpoint(model, optimizer, checkpoint)
    strategy.load_optimizer_checkpoint(restored, restored_optimizer, checkpoint)
    assert restored_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    step(model, optimizer)
    step(restored, restored_optimizer)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distributed_optimizer_load_keeps_rank_zero_broadcast():
    strategy = NativeFSDPStrategy(device_mesh=MagicMock(), enable_ep=False)
    with patch('torch.distributed.is_initialized', return_value=True):
        options = strategy._prepare_optimizer_state_dict_options(for_load=True)
    assert options.broadcast_from_rank0
