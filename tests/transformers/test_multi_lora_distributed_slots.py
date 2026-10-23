# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real two-process CPU regression for resident Multi-LoRA slot switching."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig
from safetensors.torch import save_file

from twinkle.model.multi_lora import MultiLora
from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel


def _wrapper(module, manager, wrap_model):
    wrapper = object.__new__(MultiLoraTransformersModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = module
    wrapper.multi_adapter = manager
    wrapper._model_wrapped = False
    wrapper.optimizer_group = {}
    wrapper._enable_expert_parallel = False
    wrapper.sp_strategy = None
    wrapper.strategy = SimpleNamespace(
        capture_pre_ep_state_if_needed=lambda *args, **kwargs: None,
        wrap_model=wrap_model,
        unwrap_model=lambda model: model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
    # This linear model does not use expert/sequence parallelism or GDN patches.
    for name in ('_maybe_apply_expert_parallel', '_ensure_sp_strategy', '_ensure_gdn_padding_free_patch'):
        object.__setattr__(wrapper, name, lambda: None)
    return wrapper


def _distributed_slots(rank, rendezvous, mode):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.set_num_threads(1)
        # DDP must synchronize initialization even when workers have different RNG states.
        torch.manual_seed(42 + rank)
        config = LoraConfig(r=2, lora_alpha=2, target_modules=['0'])
        manager = MultiLora(max_loras=2, max_r=2)
        module = manager.patch(torch.nn.Sequential(torch.nn.Linear(4, 4)), target_modules=['0'], lora_config=config)
        manager.save_initial_weights()
        manager.reset_adapter_status()
        manager.acquire_lora('first', config)
        wrapper = _wrapper(module, manager, lambda model: torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True))
        wrapper._lazy_wrap_model()
        assert not any(param.requires_grad for name, param in module.named_parameters() if '.base_layer.' in name)

        def synchronized_weights(tenant):
            state = manager.get_state_dict(tenant)
            flat = torch.cat([value.detach().flatten() for value in state.values()])
            copies = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(copies, flat)
            torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=1e-7)
            return {key: value.detach().clone() for key, value in state.items()}

        for tenant in (('first', 'second') if mode == 'switch' else ('first', 'reused')):
            if tenant == 'second':
                manager.acquire_lora(tenant, config)
            elif tenant == 'reused':
                manager.release_lora('first')
                manager.acquire_lora(tenant, config)
            initial = synchronized_weights(tenant)
            with manager.adapter(tenant):
                parameters = [p for p in module.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(parameters, lr=.1)
                for _ in range(3):
                    optimizer.zero_grad()
                    # Different local batches expose missing gradient synchronization.
                    wrapper.model(torch.ones(2, 4) * (rank + 1)).square().mean().backward()
                    optimizer.step()
            trained = synchronized_weights(tenant)
            assert any(not torch.equal(initial[name], value) for name, value in trained.items())
            with manager.adapter(tenant), torch.no_grad():
                fixed = wrapper.model(torch.ones(2, 4)).detach().clone()
                # Rank 0 checkpoint must preserve predictions on every DP replica.
                checkpoint = [trained if rank == 0 else None]
                dist.broadcast_object_list(checkpoint, src=0)
                manager.set_state_dict(tenant, checkpoint[0])
                torch.testing.assert_close(wrapper.model(torch.ones(2, 4)), fixed, rtol=0, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['switch', 'reuse'])
def test_all_slots_sync_across_dp_ranks_after_switch_and_reuse(tmp_path, mode):
    mp.spawn(_distributed_slots, args=(str(tmp_path / 'rendezvous'), mode), nprocs=2, join=True)


@pytest.mark.parametrize('options', [{'find_unused_parameters': False}, {'static_graph': True}])
def test_multi_lora_rejects_static_ddp_reducers(options, monkeypatch):
    monkeypatch.setattr(MultiLoraTransformersModel, '_try_init_process_group', lambda _: None)
    with pytest.raises(ValueError, match='Multi-LoRA DDP requires'):
        MultiLoraTransformersModel(ddp_config=options)


@pytest.mark.parametrize('reshard_after_forward', [None, False, 2])
def test_multi_lora_rejects_retained_or_partial_fsdp_views(reshard_after_forward, monkeypatch):
    monkeypatch.setattr(MultiLoraTransformersModel, '_try_init_process_group', lambda _: None)

    def unexpected_strategy(*args):
        raise AssertionError('Invalid FSDP config reached strategy initialization')

    monkeypatch.setattr(MultiLoraTransformersModel, '_decide_strategy', unexpected_strategy)
    with pytest.raises(ValueError, match='Multi-LoRA FSDP requires reshard_after_forward=True'):
        MultiLoraTransformersModel(fsdp_config={'reshard_after_forward': reshard_after_forward})


def test_checkpoint_as_first_operation_preserves_fresh_slot_reset_basis(tmp_path):
    config = LoraConfig(r=2, lora_alpha=2, target_modules=['0'])
    manager = MultiLora(max_loras=1, max_r=2)
    module = manager.patch(torch.nn.Sequential(torch.nn.Linear(4, 4)), target_modules=['0'], lora_config=config)
    manager.save_initial_weights()
    manager.acquire_lora('first', config)
    initial = {name: value.detach().clone() for name, value in manager.get_state_dict('first').items()}
    checkpoint = {name: torch.full_like(value, 2) for name, value in initial.items()}
    save_file(checkpoint, str(tmp_path / 'adapter_model.safetensors'))
    wrapper = _wrapper(module, manager, lambda model: model)
    wrapper.optimizer_group = {'first': SimpleNamespace(optimizer=None)}
    wrapper.load(str(tmp_path), adapter_name='first')
    for name, value in manager.get_state_dict('first').items():
        torch.testing.assert_close(value, checkpoint[name])
    # Further operations must not replace the initial reset basis with trained weights.
    wrapper._lazy_wrap_model()
    manager.release_lora('first')
    manager.acquire_lora('replacement', config)
    for name, value in manager.get_state_dict('replacement').items():
        torch.testing.assert_close(value, initial[name])


def test_slot_registration_never_unfreezes_base_parameters_with_lora_in_their_name():
    from collections import OrderedDict
    config = LoraConfig(r=2, target_modules=['lora_projection'])
    manager = MultiLora(max_loras=2, max_r=2)
    module = manager.patch(torch.nn.Sequential(OrderedDict([
        ('lora_projection', torch.nn.Linear(4, 4)),
    ])), target_modules=['lora_projection'], lora_config=config)
    base = {name: parameter for name, parameter in module.named_parameters() if '.base_layer.' in name}
    assert base
    # Respect an explicitly frozen base even when its module name contains lora_.
    for parameter in base.values():
        parameter.requires_grad_(False)
    manager.reset_adapter_status()
    mask = {name: parameter.requires_grad for name, parameter in module.named_parameters()}
    with pytest.raises(RuntimeError, match='wrapper failed'):
        with manager.trainable_slots():
            assert all(parameter.requires_grad for _, parameter in manager._named_slot_parameters(module))
            assert not any(parameter.requires_grad for parameter in base.values())
            raise RuntimeError('wrapper failed')
    assert {name: parameter.requires_grad for name, parameter in module.named_parameters()} == mask
    manager.acquire_lora('first', config)
    wrapper = _wrapper(module, manager, lambda model: model)
    for disable_lora in (False, True):
        with wrapper._adapter_context('first', disable_lora=disable_lora):
            assert not any(parameter.requires_grad for parameter in base.values())
        assert not any(parameter.requires_grad for parameter in base.values())
