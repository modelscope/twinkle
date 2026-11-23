# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect115 permanent gate: receive_weights_ema must update the LIVE sharded reference.

Torch.to_local_tensor gathers a DTensor to a FULL copy (full_tensor(), not to_local()). The old
receive_weights_ema built local_state from _get_local_state_dict() and blended into those gathered
throwaway copies, so a sharded (FSDP2) reference's live parameters never changed -- a vacuous EMA
(CPU probe: 0/15 live params changed). The fix blends into the live local shard, slicing the
received full tensor to this rank's placement (redistribute against the target's own placements).

PLAN (case | input | expected | failure it kills):
1 sharded_live_blend | real 2-rank gloo FSDP2 sharded bare ref + real receive_weights_ema over known
  full tensors | each live local shard == 0.6*before + 0.4*received_shard (hand oracle, dim-0 slice
  per rank) | the exact defect115 vacuous EMA (live params untouched)
2 replicated_control | unsharded bare ref, same feed | live param == 0.6*before + 0.4*received_full |
  over-correcting the replicated path that already worked
3 reverse-verify (procedure) | restore the pre-fix receiver | case 1 RED (live unchanged), case 2
  GREEN | a gate that passes without the fix proves nothing

Oracle independence: the expected blend is hand-computed from the pre-blend live shard and the
received tensor sliced by the rank's dim-0 geometry, never through the receiver's own math. Only
the transport is a seam; the real NCCL pairing + KL sawtooth are the GPU EMA e2e sign-off.
"""
import os
import time
import zlib
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch


class _Feed:
    def __init__(self, pairs):
        self._pairs = pairs

    async def receive_weights(self):
        for name, tensor in self._pairs:
            yield name, tensor


def _shard(ref):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    # Explicit CPU mesh: current torch's fully_shard default mesh picks cuda whenever CUDA is
    # visible, even over a gloo PG -- the cuda collective on gloo then SIGSEGVs. This gate's
    # contract is the CPU/gloo blend math, so pin the mesh to the PG's actual device.
    cpu_mesh = init_device_mesh('cpu', (2,))
    for layer in ref.model.layers:
        fully_shard(layer, mesh=cpu_mesh)
    fully_shard(ref.model.embed_tokens, mesh=cpu_mesh)
    fully_shard(ref.lm_head, mesh=cpu_mesh)
    fully_shard(ref, mesh=cpu_mesh)


def _gate(rank, directory):
    import torch.distributed as dist
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from twinkle import DeviceMesh, Platform
    from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy
    from twinkle.model.transformers.transformers import TransformersModel
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
            cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                              tie_word_embeddings=False)

            def _shell(model):
                tm = object.__new__(TransformersModel)
                torch.nn.Module.__init__(tm)
                tm.strategy = strategy
                tm.model = model
                tm.hf_config = cfg
                tm._checkpoint_engine_backend = 'nccl'
                tm._checkpoint_engine_key = ('nccl', 'default')
                return tm

            def _feed(tm, pairs):
                tm._checkpoint_engines = {('nccl', 'default'): _Feed(pairs)}
                tm.receive_weights_ema(alpha=0.6)

            # case 1: sharded ref live local shards must blend
            torch.manual_seed(7)
            ref = Qwen2ForCausalLM(cfg)
            _shard(ref)
            tm = _shell(ref)
            before, pairs = {}, []
            for name, p in ref.named_parameters():
                local = p.to_local() if hasattr(p, 'to_local') else p
                before[name] = local.detach().clone()
                full = p.full_tensor() if hasattr(p, 'full_tensor') else p.detach()
                g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                pairs.append((name, torch.randn(full.shape, generator=g) * 0.1 + 0.3))
            _feed(tm, pairs)
            recv = dict(pairs)
            for name, p in ref.named_parameters():
                local = (p.to_local() if hasattr(p, 'to_local') else p).detach()
                r = recv[name]
                if hasattr(p, 'placements') and p.placements[0].is_shard():
                    dim = p.placements[0].dim
                    n = r.shape[dim]
                    per = (n + 1) // 2
                    r = r.narrow(dim, rank * per, min(per, n - rank * per))
                expected = before[name] * 0.6 + r * 0.4
                torch.testing.assert_close(local, expected, rtol=1e-6, atol=1e-7)
            dist.barrier()

            # case 2: replicated ref control (plain params) must still blend
            torch.manual_seed(7)
            plain = Qwen2ForCausalLM(cfg)
            tm2 = _shell(plain)
            before2, pairs2 = {}, []
            for name, p in plain.named_parameters():
                before2[name] = p.detach().clone()
                g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                pairs2.append((name, torch.randn(p.shape, generator=g) * 0.1 + 0.3))
            _feed(tm2, pairs2)
            recv2 = dict(pairs2)
            for name, p in plain.named_parameters():
                expected = before2[name] * 0.6 + recv2[name] * 0.4
                torch.testing.assert_close(p.detach(), expected, rtol=1e-6, atol=1e-7)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ema_receiver_updates_live_sharded_ref(tmp_path):
    context = torch.multiprocessing.spawn(_gate, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('EMA receiver live-shard gate did not finish within 180 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
