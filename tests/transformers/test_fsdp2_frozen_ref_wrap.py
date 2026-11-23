# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect110 permanent gate: a frozen FSDP2 reference must wrap WITHOUT an optimizer.

The separate KL reference (GRPO sync_ref_model / full-param separate-ref) is forward-only: it has
no optimizer and never will. ``Accelerator.prepare``'s FSDP2 guard refuses model-only calls
("a model and optimizer must be passed together"), so the reference died at its first
``_lazy_wrap_model`` before any KL was computed. ``wrap_model`` must route the model-only FSDP2
case through ``_prepare_fsdp2`` (every optimizer step there is conditional) instead.

PLAN (case | input | expected | failure it kills):
1 frozen model-only wrap | real 2-rank gloo FSDP2, requires_grad_(False) Qwen2, wrap_model(model)
  | FSDP-managed model, forward runs | the accelerate model-only refusal (defect110 exact old red)
2 trainable model+optimizer control | same mesh, wrap_model(model, optimizer) | prepared via
  prepare(), optimizer params are the wrapped model's objects | regression on the training path
  the fix must not touch
3 frozen forward value | wrapped frozen model logits vs the pre-wrap eager logits | identical
  (sharding is lossless) | a wrap that silently changes reference outputs (KL anchor corruption)

Reverse-verify: removing the model-only branch (always prepare()) turns case 1 RED with the
accelerate ValueError; cases 2/3 stay GREEN.
"""
import os
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch


def _wrap_frozen_reference(rank, directory):
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from twinkle import DeviceMesh, Platform
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    from twinkle.utils.framework import Torch

    os.environ.update(RANK=str(rank), LOCAL_RANK='0', WORLD_SIZE='2', ACCELERATE_USE_CPU='true',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT='29500')
    os.environ.pop('LOCAL_WORLD_SIZE', None)
    dist.init_process_group('gloo', init_method=f'file://{directory}/pg', rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        mesh = DeviceMesh.from_sizes(world_size=2, fsdp_size=2, device_type='cpu')
        with patch.object(Platform, 'get_local_device', return_value='cpu'), \
                patch.object(Platform, 'device_backend', return_value='gloo'), \
                patch.object(Torch, '_device_module', return_value=None):
            strategy = AccelerateStrategy(device_mesh=mesh, mixed_precision='bf16')
            # accelerate only enters its FSDP branch under MULTI_GPU; on a CPU gloo PG the launcher
            # state must be set by hand. The wrap itself (fully_shard, all-gather) runs for real.
            from accelerate.utils import DistributedType
            strategy.accelerator.state.distributed_type = DistributedType.FSDP
            strategy.accelerator.state.fsdp_plugin = strategy._fsdp_config_from_device_mesh(mesh, None, False)
            # accelerate's fsdp2_prepare_model shards on state.device_mesh; without one, fully_shard
            # builds its DEFAULT mesh, which current torch pins to cuda whenever CUDA is visible --
            # moving this CPU/gloo reference to cuda:0. Hand it the CPU mesh the gate runs on.
            from accelerate import ParallelismConfig
            from torch.distributed.device_mesh import init_device_mesh
            strategy.accelerator.state.parallelism_config = ParallelismConfig(dp_shard_size=2)
            strategy.accelerator.state.device_mesh = init_device_mesh('cpu', (2,), mesh_dim_names=('dp_shard_cp',))
            config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
            torch.manual_seed(7)
            model = Qwen2ForCausalLM(config).requires_grad_(False)
            inputs = torch.randint(0, 32, (1, 4))
            with torch.no_grad():
                eager = model.to(torch.bfloat16)(inputs).logits

            wrapped = strategy.wrap_model(model)
            assert isinstance(wrapped, FSDPModule) or any(isinstance(m, FSDPModule) for m in wrapped.modules())
            with torch.no_grad():
                actual = wrapped(inputs).logits
            torch.testing.assert_close(actual, eager, rtol=0, atol=0)

            torch.manual_seed(7)
            trainable = Qwen2ForCausalLM(config)
            optimizer = torch.optim.AdamW(trainable.parameters(), lr=1e-4)
            prepared = strategy.wrap_model(trainable, optimizer)
            wrapped_trainable, prepared_optimizer = prepared[0], prepared[1]
            model_params = {id(p) for p in wrapped_trainable.parameters()}
            assert all(id(p) in model_params for group in prepared_optimizer.param_groups for p in group['params'])
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_frozen_fsdp2_reference_wraps_without_optimizer(tmp_path):
    context = torch.multiprocessing.spawn(_wrap_frozen_reference, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 120
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('frozen FSDP2 reference wrap did not finish within 120 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
