# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect111 permanent gate: the paired-EMA sender must emit bare-ref names, LoRA merged locally.

The EMA reference is a bare (non-PEFT) frozen model. A LoRA policy's ``state_dict()`` carries
``base_model.model.*`` prefixes plus ``lora_A/lora_B`` keys, so the old ``_get_local_weight_generator``
yielded names the receiver's ``local_state[name]`` lookup could never match -- the first sync died
with KeyError on the first key (GPU e2e: Qwen3.5-4B visual.patch_embed). The fix folds the adapter
into the base weight in the transport's tensor space (``Torch.to_local_tensor`` gathers a DTensor to
its full tensor, the same call the receiver makes) and yields canonical names.

PLAN (case | input | expected | failure it kills):
1 lora_name_match | real 2-rank gloo FSDP2 LoRA policy + bare ref, nonzero lora_B | sent name set ==
  ref canonical keys, per-name local shape matches, values bit-equal an independent unsharded
  merged oracle sliced to this rank | the exact defect111 KeyError (0/15 name matches) AND a
  vacuous/wrong merge (names right, values left unmerged or mis-sharded)
2 full_identity | plain (non-PEFT) policy + bare ref, same shard layout | names == ref keys, local
  values identical | over-normalizing the full-param path that already worked
3 tied_once | tie_word_embeddings=True, unsharded LoRA policy | lm_head.weight absent, embed sent
  exactly once | tied double-send (the receiver would EMA the shared storage twice)
4 merge_restore | policy local-shard snapshot before/after exhausting the generator | bit-equal |
  an in-place merge leaking into the live training weights
5 receiver_blend | real receive_weights_ema over the sender's actual pairs, alpha=0.6 | ref ==
  0.6*ref_before + 0.4*sent element-wise, and the adapter delta actually propagated | EMA
  direction/alpha errors, and a sync that silently sends unmerged base weights
6 reverse-verify (procedure, not a test) | restore the pre-fix generator | cases 1/3/5 RED, 2/4
  GREEN | a gate that passes without the fix proves nothing

Oracle independence: the merged reference values come from a second UNSHARDED PeftModel merged by
peft's own merge_adapter and read by MODULE NAVIGATION (get_submodule + base_layer), never through
the sender's name transform; the blend oracle is hand-computed. Only the transport is a seam (an
in-process async pair feed) -- the real NCCL pairing is covered by the GPU EMA e2e arm.
"""
import os
import time
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch


class _PairFeed:
    """Transport seam: replays the sender's captured (name, tensor) pairs as the receive stream."""

    def __init__(self, pairs):
        self._pairs = pairs

    async def receive_weights(self):
        for name, tensor in self._pairs:
            yield name, tensor


def _shard_fsdp2(model):
    from peft import PeftModel
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    # Explicit CPU mesh: current torch's fully_shard default mesh picks cuda whenever CUDA is
    # visible, even over a gloo PG -- the cuda collective on gloo then SIGSEGVs. This gate's
    # contract is the CPU/gloo transport math, so pin the mesh to the PG's actual device.
    cpu_mesh = init_device_mesh('cpu', (2,))
    base = model.base_model.model if isinstance(model, PeftModel) else model
    for layer in base.model.layers:
        fully_shard(layer, mesh=cpu_mesh)
    fully_shard(base.model.embed_tokens, mesh=cpu_mesh)
    fully_shard(base.lm_head, mesh=cpu_mesh)
    fully_shard(model, mesh=cpu_mesh)


def _ema_gate(rank, directory):
    import torch.distributed as dist
    import torch.nn as nn
    from peft import LoraConfig, PeftModel, get_peft_model
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
            # 'bf16' satisfies both Accelerator and the FSDP plugin; params stay fp32 because the
            # gate shards with raw fully_shard (no wrap_model cast), keeping the merge oracle bit-exact.
            strategy = AccelerateStrategy(device_mesh=mesh, mixed_precision='bf16')

            def _shell(model, config):
                tm = object.__new__(TransformersModel)
                torch.nn.Module.__init__(tm)
                tm.strategy = strategy
                tm.model = model
                tm.hf_config = config
                return tm

            def _config(tied):
                return Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                                   tie_word_embeddings=tied)

            lora = LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
                              lora_dropout=0.0, bias='none')
            cfg = _config(tied=False)
            torch.manual_seed(7)
            policy = get_peft_model(Qwen2ForCausalLM(cfg), lora)
            # Zero-init lora_B would make the merge an identity at birth; perturb it so a vacuous
            # (unmerged) sender is distinguishable from a correct one by value, not just by name.
            for name, param in policy.named_parameters():
                if 'lora_B' in name:
                    nn.init.normal_(param, std=0.02)
            lora_sd = {k: v.detach().clone() for k, v in policy.state_dict().items() if 'lora_' in k}

            # Independent oracle: unsharded copy, peft's own merge, values read by module navigation.
            torch.manual_seed(7)
            oracle = get_peft_model(Qwen2ForCausalLM(cfg), lora)
            oracle.load_state_dict(lora_sd, strict=False)
            oracle.merge_adapter()
            torch.manual_seed(7)
            ref = Qwen2ForCausalLM(cfg)
            oracle_merged = {}
            for key in ref.state_dict():
                prefix, leaf = key.rsplit('.', 1)
                module = oracle.base_model.model.get_submodule(prefix)
                base_module = module.base_layer if hasattr(module, 'base_layer') else module
                oracle_merged[key] = getattr(base_module, leaf).detach().clone()
            oracle.unmerge_adapter()

            _shard_fsdp2(policy)
            _shard_fsdp2(ref)
            policy_shell = _shell(policy, cfg)
            ref_shell = _shell(ref, cfg)
            ref_local = ref_shell._get_local_state_dict()
            world = 2

            def _oracle_local(key, full):
                shard = ref_local[key]
                if hasattr(shard, 'placements') and shard.placements[0].is_shard():
                    dim = shard.placements[0].dim
                    n = full.shape[dim]
                    per = (n + world - 1) // world
                    return full.narrow(dim, rank * per, min(per, n - rank * per))
                return full

            # case 1: lora_name_match (names + shapes + bit-level merged values)
            pairs = list(policy_shell._get_local_weight_generator())
            sent = dict(pairs)
            assert len(sent) == len(pairs), 'duplicate names in EMA stream'
            assert set(sent) == set(ref_local), (
                f'sent names != ref keys; missing={sorted(set(ref_local) - set(sent))[:3]} '
                f'extra={sorted(set(sent) - set(ref_local))[:3]}')
            assert not any('lora_' in k or k.startswith('base_model.model.') for k in sent)
            for key, tensor in sent.items():
                assert tensor.shape == ref_local[key].shape, f'shard shape mismatch for {key}'
                torch.testing.assert_close(tensor, _oracle_local(key, oracle_merged[key]), rtol=0, atol=0)
            dist.barrier()

            # case 2: full_identity (plain policy path unchanged)
            torch.manual_seed(7)
            plain = Qwen2ForCausalLM(cfg)
            _shard_fsdp2(plain)
            plain_sent = dict(_shell(plain, cfg)._get_local_weight_generator())
            assert set(plain_sent) == set(ref_local)
            for key, tensor in plain_sent.items():
                assert torch.equal(tensor, ref_local[key]), f'full-param identity broken for {key}'
            dist.barrier()

            # case 3: tied_once (name filtering is placement-independent; unsharded keeps the
            # tied+FSDP2 construction question out of this case's failure surface)
            tied_cfg = _config(tied=True)
            torch.manual_seed(7)
            tied_policy = get_peft_model(Qwen2ForCausalLM(tied_cfg), lora)
            tied_sent = [name for name, _ in _shell(tied_policy, tied_cfg)._get_local_weight_generator()]
            assert 'lm_head.weight' not in tied_sent, 'tied head double-sent'
            assert tied_sent.count('model.embed_tokens.weight') == 1
            assert len(tied_sent) == len(set(tied_sent))
            dist.barrier()

            # case 4: merge_restore (generator must not mutate the live policy shards)
            before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            list(policy_shell._get_local_weight_generator())
            for key, value in policy.state_dict().items():
                assert torch.equal(value, before[key]), f'generator mutated live weight {key}'
            dist.barrier()

            # case 5: receiver_blend (real receive_weights_ema math over the real sender pairs).
            # The ref is UNSHARDED so the receiver mutates live params: a sharded ref's
            # _get_local_state_dict gathers throwaway full copies, and that gather is a collective
            # which must not run inside the receive thread. The blend code path is the real one.
            torch.manual_seed(7)
            blend_ref = Qwen2ForCausalLM(cfg)
            blend_shell = _shell(blend_ref, cfg)
            blend_local = blend_shell._get_local_state_dict()
            ref_before = {k: v.detach().clone() for k, v in blend_local.items()}
            blend_shell._checkpoint_engine_backend = 'nccl'
            blend_shell._checkpoint_engine_key = ('nccl', 'default')
            blend_shell._checkpoint_engines = {('nccl', 'default'): _PairFeed(pairs)}
            blend_shell.receive_weights_ema(alpha=0.6)
            for key, tensor in sent.items():
                expected = ref_before[key] * 0.6 + tensor * 0.4
                torch.testing.assert_close(blend_local[key], expected, rtol=1e-6, atol=1e-7)
            merged_target = next(k for k in sent if k.endswith('q_proj.weight'))
            unmerged = ref_before[merged_target]
            assert not torch.equal(blend_local[merged_target], unmerged * 0.6 + unmerged * 0.4), (
                'adapter delta did not propagate -- vacuous sync')
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_ema_lora_local_merge_gate(tmp_path):
    context = torch.multiprocessing.spawn(_ema_gate, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail('EMA LoRA local-merge gate did not finish within 180 seconds')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
