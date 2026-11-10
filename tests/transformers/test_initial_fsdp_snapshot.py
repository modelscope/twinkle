# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


def _make_checkpoint(directory, ep_size):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    if ep_size > 1:
        transformers = pytest.importorskip('transformers')
        if not hasattr(transformers, 'Qwen3_5MoeConfig'):
            pytest.skip('Qwen3.5 MoE requires a Transformers version with Qwen3_5MoeConfig')
        config = transformers.Qwen3_5MoeConfig(
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                             moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2,
                             layer_types=['full_attention']),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                               out_hidden_size=16, patch_size=2, temporal_patch_size=2,
                               spatial_merge_size=2))
        model = transformers.Qwen3_5MoeForConditionalGeneration(config)
    else:
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
        model = Qwen2ForCausalLM(config)
    model.save_pretrained(directory)
    tokenizer = Tokenizer(WordLevel({f'w{i}': i for i in range(32)}, unk_token='w0'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='w0', pad_token='w0').save_pretrained(directory)


def _save_initial_snapshot(rank, directory, ep_size):
    import torch.distributed as dist
    from peft import LoraConfig
    from safetensors.torch import load_file

    from twinkle import DeviceGroup, DeviceMesh, Platform, initialize
    from twinkle.model.state_dict_utils import is_lora_state_key, is_saveable_adapter_key, normalize_adapter_keys
    from twinkle.model.transformers import TransformersModel
    from twinkle.model.transformers.strategy import native_fsdp
    from twinkle.model.transformers.strategy.native_fsdp import (
        _detect_ep_expert_names,
        _ep_expert_state_dict_gather_dim,
        _resolve_full_state_source_key,
    )

    world_size = 4 if ep_size > 1 else 2
    os.environ.update(RANK=str(rank), LOCAL_RANK='0', WORLD_SIZE=str(world_size))
    os.environ.pop('LOCAL_WORLD_SIZE', None)
    dist.init_process_group('gloo', init_method=f'file://{directory}/pg', rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=20))
    try:
        mesh = DeviceMesh.from_sizes(world_size=world_size, fsdp_size=world_size, device_type='cpu',
                                     ep_size=ep_size, ep_fsdp_size=world_size // ep_size)
        with patch.object(Platform, 'get_local_device', return_value='cpu'), \
                patch.object(Platform, 'device_backend', return_value='gloo'):
            initialize(mode='local', global_device_mesh=mesh,
                       groups=[DeviceGroup(name='default', ranks=list(range(world_size)), device_type='cpu')])
            # Ray actors each expose one device, so every actor has LOCAL_RANK=0.
            os.environ['TWINKLE_MODE'] = 'ray'
            wrapper = TransformersModel(
                model_id=directory, device_mesh=mesh, strategy='native_fsdp', memory_efficient_init=True,
                mixed_precision='no', gradient_checkpointing=False, dtype=torch.float32,
                fsdp_config=({'expert_parallel': {'enabled': True, 'ep_size': ep_size}} if ep_size > 1 else {}))
            checkpoint_holder = [{name: tensor.detach().cpu().clone()
                                  for name, tensor in wrapper.model.state_dict().items()} if rank == 0 else None]
            dist.broadcast_object_list(checkpoint_holder, src=0)
            checkpoint = checkpoint_holder[0]
            lora = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'])
            if ep_size > 1:
                lora.target_parameters = ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']
            torch.manual_seed(31 + rank)
            wrapper.add_adapter_to_model('default', lora)
            wrapper.set_optimizer('AdamW', lr=1e-4)
            assert not wrapper._model_wrapped
            assert any(parameter.is_meta for parameter in wrapper.model.parameters()) == (rank != 0)

            experts = _detect_ep_expert_names(wrapper.model)
            expected_shapes = {}
            expected_replicated = {}
            for name, tensor in wrapper.model.named_parameters():
                if not is_saveable_adapter_key(name) or '.default.' not in name:
                    continue
                shape = list(tensor.shape)
                if name in experts:
                    shape[_ep_expert_state_dict_gather_dim(name)] *= ep_size
                elif rank == 0:
                    expected_replicated[name] = tensor.detach().cpu().clone()
                expected_shapes[name] = tuple(shape)
            expected_shapes = normalize_adapter_keys(expected_shapes, 'default')
            expected_replicated = normalize_adapter_keys(expected_replicated, 'default')

            # This is the first model use: no forward may initialize FSDP before v0.
            # EP-local weights must stay on meta until FSDP has described their
            # shards; moving all experts first can exceed device memory.
            with patch.object(native_fsdp, '_move_ep_module_to_device',
                              side_effect=AssertionError('EP weights must be sharded before materialization')):
                path = wrapper.save('v0', output_dir=str(Path(directory) / 'snapshots'),
                                    adapter_name='default', save_optimizer=False)
            assert wrapper._model_wrapped
            assert all(not parameter.is_meta for parameter in wrapper.model.parameters())
            optimizer = wrapper.optimizer_group['default'].optimizer
            assert all(not parameter.is_meta for group in optimizer.param_groups for parameter in group['params'])
            # Deferred placement must preserve both replicated and expert weights.
            for name, tensor in wrapper.model.state_dict().items():
                if is_lora_state_key(name):
                    continue
                actual = tensor.full_tensor() if hasattr(tensor, 'full_tensor') else tensor
                expected = checkpoint[_resolve_full_state_source_key(name, checkpoint)]
                if name in experts:
                    ep_rank = wrapper.strategy.ep_fsdp_device_mesh['ep'].get_local_rank()
                    expected = expected.chunk(ep_size, dim=0)[ep_rank]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dist.barrier()
            saved = load_file(Path(path) / 'adapter_model.safetensors')
            assert {name: tuple(tensor.shape) for name, tensor in saved.items()} == expected_shapes
            assert all(torch.isfinite(tensor).all() for tensor in saved.values())
            for name, tensor in expected_replicated.items():
                torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)

            # A later publication must reuse the wrapped model and optimizer.
            next_path = wrapper.save('v1', output_dir=str(Path(directory) / 'snapshots'),
                                     adapter_name='default', save_optimizer=False)
            assert wrapper.optimizer_group['default'].optimizer is optimizer
            dist.barrier()
            next_saved = load_file(Path(next_path) / 'adapter_model.safetensors')
            assert next_saved.keys() == saved.keys()
            for name, tensor in saved.items():
                torch.testing.assert_close(next_saved[name], tensor, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('ep_size', [1, 2, 4], ids=['fsdp2', 'fsdp4-ep2-epfsdp2', 'fsdp4-ep4-epfsdp1'])
def test_initial_adapter_snapshot_materializes_all_ranks(tmp_path, ep_size):
    _make_checkpoint(tmp_path, ep_size)
    world_size = 4 if ep_size > 1 else 2
    context = torch.multiprocessing.spawn(_save_initial_snapshot, args=(str(tmp_path), ep_size),
                                          nprocs=world_size, join=False)
    try:
        deadline = time.monotonic() + 60
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('Initial FSDP adapter snapshot did not finish within 60 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
