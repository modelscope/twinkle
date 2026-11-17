# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from twinkle.model.transformers.strategy import native_fsdp
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
from twinkle.model.transformers.transformers import TransformersModel


@pytest.mark.parametrize('rank', range(4))
def test_ray_loading_source_uses_node_membership_instead_of_device_index(rank):
    # Interleaved ranks on two nodes, all with their own visible device 0.
    hosts = ['node-a', 'node-b', 'node-a', 'node-b']
    local_ranks = [index for index, host in enumerate(hosts) if host == hosts[rank]]

    def gather(result, host):
        assert host == hosts[rank]
        result[:] = hosts

    with patch.dict(os.environ, {'TWINKLE_MODE': 'ray', 'LOCAL_RANK': '0', 'LOCAL_WORLD_SIZE': '8'}), \
            patch.object(native_fsdp.dist, 'get_rank', return_value=rank), \
            patch.object(native_fsdp.dist, 'get_world_size', return_value=4), \
            patch.object(native_fsdp.socket, 'gethostname', return_value=hosts[rank]), \
            patch.object(native_fsdp.dist, 'all_gather_object', side_effect=gather):
        assert native_fsdp._get_local_rank_info() == (rank, 4, local_ranks[0], local_ranks)
        strategy = NativeFSDPStrategy()
        assert strategy.is_pretrained_loading_source() == (rank == local_ranks[0])
        assert os.environ['LOCAL_RANK'] == '0'


@pytest.mark.parametrize('rank', range(4))
def test_torchrun_loading_source_keeps_local_rank_topology(rank):
    with patch.dict(os.environ, {'TWINKLE_MODE': 'local', 'LOCAL_RANK': str(rank % 2), 'LOCAL_WORLD_SIZE': '2'}), \
            patch.object(native_fsdp.dist, 'get_rank', return_value=rank), \
            patch.object(native_fsdp.dist, 'get_world_size', return_value=4), \
            patch.object(native_fsdp.dist, 'all_gather_object') as gather:
        source = rank - rank % 2
        assert native_fsdp._get_local_rank_info() == (rank, 4, source, [source, source + 1])
        gather.assert_not_called()


@pytest.mark.parametrize('rank', [0, 1])
def test_pre_ep_capture_only_keeps_the_loading_sources_real_weights(rank):
    strategy = NativeFSDPStrategy(device_mesh=SimpleNamespace(ep_size=1), memory_efficient_init=True,
                                  enable_ep=False)
    with torch.device('cpu' if rank == 0 else 'meta'):
        model = torch.nn.Linear(4, 4)
    with patch.object(native_fsdp, '_get_local_rank_info', return_value=(rank, 2, 0, [0, 1])):
        strategy.capture_pre_ep_state_if_needed(model, enable_ep=True)
    assert strategy._pre_ep_state_captured
    if rank == 0:
        assert strategy._rank0_pre_ep_full_state_dict.keys() == model.state_dict().keys()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(strategy._rank0_pre_ep_full_state_dict[name], tensor)
    else:
        assert strategy._rank0_pre_ep_full_state_dict == {}


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
                              model_arch={'language_model': [], 'vision_tower': [], 'aligner': []},
                              _model_wrapped=False, _enable_expert_parallel=False, sp_strategy=None,
                              _maybe_apply_expert_parallel=lambda: None, _ensure_sp_strategy=lambda: None,
                              _ensure_gdn_padding_free_patch=lambda: None,
                              _rebind_lr_scheduler=TransformersModel._rebind_lr_scheduler,
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


@pytest.fixture
def head_sharding():
    strategy = NativeFSDPStrategy(device_mesh=MagicMock(), mixed_precision='bf16', enable_ep=False)
    with patch('twinkle.model.transformers.strategy.native_fsdp._build_fsdp_mesh',
               return_value=MagicMock()), \
         patch('twinkle.model.transformers.strategy.native_fsdp.fully_shard') as fully_shard:
        yield strategy, fully_shard


def test_builtin_score_head_remains_in_root_fsdp_group(head_sharding):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.score = torch.nn.Linear(4, 2).to(torch.bfloat16)
    strategy, fully_shard = head_sharding
    strategy.wrap_model(model)

    fully_shard.assert_called_once()
    assert fully_shard.call_args.args[0] is model
    assert fully_shard.call_args.kwargs['mp_policy'].param_dtype is torch.bfloat16


def test_nonmaster_save_ignores_empty_gathered_optimizer_state(tmp_path):
    strategy = NativeFSDPStrategy(device_mesh=MagicMock(), enable_ep=False)
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with patch('twinkle.model.transformers.strategy.native_fsdp.Platform.is_master', return_value=False), \
            patch('torch.distributed.checkpoint.state_dict.get_optimizer_state_dict', return_value={}) as get_state, \
            patch('torch.save') as save:
        strategy.save_optimizer_checkpoint(model, optimizer, str(tmp_path / 'optimizer.pt'))

    get_state.assert_called_once()
    save.assert_not_called()


def test_optimizer_restore_accepts_lazy_state_for_unused_parameters(tmp_path):
    strategy = NativeFSDPStrategy(device_mesh=MagicMock(), enable_ep=False)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model[0](torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    with patch('twinkle.model.transformers.strategy.native_fsdp.Platform.is_master', return_value=True):
        strategy.save_optimizer_checkpoint(model, optimizer, str(tmp_path / 'optimizer.pt'))
    state = torch.load(tmp_path / 'optimizer.pt', weights_only=True)
    assert state['state']['1.weight'] == {}

    cold = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    cold_optimizer = torch.optim.Adam(cold.parameters(), lr=1e-3)
    prepare_options = strategy._prepare_optimizer_state_dict_options

    def cpu_options(*, for_load):
        options = prepare_options(for_load=for_load)
        options.broadcast_from_rank0 = False
        return options

    with patch.object(strategy, '_prepare_optimizer_state_dict_options', side_effect=cpu_options), \
            patch('twinkle.model.transformers.strategy.native_fsdp.Platform.is_master', return_value=True):
        strategy.load_optimizer_checkpoint(cold, cold_optimizer, str(tmp_path / 'optimizer.pt'))
    assert cold_optimizer.state[cold[0].weight]['step'].item() == 1
    assert not cold_optimizer.state[cold[1].weight]

    corrupted = deepcopy(state)
    del corrupted['state']['0.weight']
    torch.save(corrupted, tmp_path / 'corrupted.pt')
    with patch.object(strategy, '_prepare_optimizer_state_dict_options', side_effect=cpu_options), \
            patch('twinkle.model.transformers.strategy.native_fsdp.Platform.is_master', return_value=True), \
            pytest.raises((KeyError, RuntimeError), match='0.weight'):
        strategy.load_optimizer_checkpoint(cold, cold_optimizer, str(tmp_path / 'corrupted.pt'))


def test_nonmaster_restores_lazy_slots_from_broadcast_metadata():
    strategy = NativeFSDPStrategy(device_mesh=MagicMock(), enable_ep=False)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    def broadcast_empty_states(payload, src):
        assert src == 0
        payload[0] = {'1.weight'}

    with patch('twinkle.model.transformers.strategy.native_fsdp.Platform.is_master', return_value=False), \
            patch('twinkle.model.transformers.strategy.native_fsdp.dist.is_initialized', return_value=True), \
            patch('twinkle.model.transformers.strategy.native_fsdp.dist.broadcast_object_list',
                  side_effect=broadcast_empty_states), \
            patch('torch.distributed.checkpoint.state_dict.set_optimizer_state_dict') as set_state:
        strategy.load_optimizer_checkpoint(model, optimizer, '/unused')

    set_state.assert_called_once()
    assert optimizer.state[model[0].weight]
    assert not optimizer.state[model[1].weight]


def test_fp32_task_head_is_sharded_as_separate_precision_group(head_sharding):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.task_head = torch.nn.Linear(4, 1)
    model.task_head.keep_fp32 = True
    model.scoring_head = model.task_head
    model.standard_head = torch.nn.Linear(4, 1).to(torch.bfloat16)
    model.task_heads = {'': model.standard_head}
    strategy, fully_shard = head_sharding
    strategy.wrap_model(model)

    assert fully_shard.call_count == 2
    head_call, model_call = fully_shard.call_args_list
    assert head_call.args[0] is model.task_head
    assert head_call.kwargs['mp_policy'].param_dtype is None
    assert head_call.kwargs['mp_policy'].reduce_dtype is None
    assert model_call.args[0] is model
    assert model_call.kwargs['mp_policy'].param_dtype is torch.bfloat16
    assert model.task_head.bias.dtype is torch.float32
    assert model.standard_head.weight.dtype is torch.bfloat16

