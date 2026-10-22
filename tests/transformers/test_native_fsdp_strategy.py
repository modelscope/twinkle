# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest
import torch

from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel


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


@pytest.mark.parametrize('tenant_count', [1, 2])
def test_lazy_wrapping_rebinds_each_tenants_optimizer_without_other_parameters(tenant_count):
    original = torch.nn.ParameterDict({name: torch.nn.Parameter(torch.ones(2)) for name in ('a', 'b', 'base')})
    replacement = torch.nn.ParameterDict({name: torch.nn.Parameter(torch.ones(2)) for name in original})
    groups = {
        name: SimpleNamespace(optimizer=torch.optim.Adam([{'params': [original[name]], 'param_names': [name]}], lr=0.1))
        for name in ('a', 'b')[:tenant_count]
    }
    mesh = MagicMock()
    mesh.mesh_dim_names = None
    strategy = NativeFSDPStrategy(device_mesh=mesh, enable_ep=False)
    wrapper = SimpleNamespace(model=original, optimizer_group=groups, strategy=strategy,
                              _model_wrapped=False, _enable_expert_parallel=False, sp_strategy=None,
                              _maybe_apply_expert_parallel=lambda: None, _ensure_sp_strategy=lambda: None,
                              _ensure_gdn_padding_free_patch=lambda: None,
                              register_mm_forward_hook=lambda group: None)
    with patch.object(original, 'to', return_value=replacement), patch(
            'twinkle.model.transformers.strategy.native_fsdp.Platform.get_local_device', return_value='cpu'):
        TransformersModel._lazy_wrap_model(wrapper)
    assert wrapper.model is replacement
    for name, group in groups.items():
        params = group.optimizer.param_groups[0]['params']
        assert len(params) == 1 and params[0] is replacement[name]
        replacement[name].grad = torch.ones_like(replacement[name])
        group.optimizer.step()
        assert not torch.equal(replacement[name], original[name])
    torch.testing.assert_close(replacement['base'], torch.ones(2))
    if tenant_count == 1:
        torch.testing.assert_close(replacement['b'], torch.ones(2))
