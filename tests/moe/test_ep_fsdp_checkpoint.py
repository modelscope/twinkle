# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test EP+FSDP checkpoint save/load round-trip:
1. Load model on single device, record full state dict as reference
2. Load same model with EP+FSDP (4 devices, ep_size=2), save checkpoint
3. Load saved checkpoint on single device, compare against reference

Requirements:
  - 4 CUDA GPUs or 4 Ascend NPUs
  - Model weights accessible via QWEN3_MOE_MODEL_ID (default: Qwen/Qwen3-30B-A3B-Instruct-2507)

Launch (requires 4 accelerators; skipped automatically if fewer are available):

    pytest tests/moe/test_ep_fsdp_checkpoint.py -v -s

    # To use a local model:
    QWEN3_MOE_MODEL_ID=/path/to/model pytest tests/moe/test_ep_fsdp_checkpoint.py -v -s

Note: nproc_per_node=1 is intentional — the test internally spawns worker processes via
mp.spawn (1 for the single-device baseline and 4 for the EP+FSDP run).
"""

import numpy as np
import os
import shutil
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from transformers import AutoConfig, AutoModelForCausalLM

from twinkle.model.transformers.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.utils import DeviceMesh
from twinkle.utils.framework import Torch as torch_util
from twinkle.utils.platforms import Platform

ABS_TOL = 1e-5
REL_TOL = 1e-5


def _get_device_info():
    """Return (device_type, backend, device_count) based on available platform."""
    platform = Platform.get_platform()
    device_type = platform.device_prefix()  # 'cuda' or 'npu'
    backend = platform.device_backend()     # 'nccl' or 'hccl'
    if device_type == 'npu':
        import torch_npu  # noqa: F401
        device_count = torch.npu.device_count()
    else:
        device_count = torch.cuda.device_count()
    return device_type, backend, device_count


def _set_device(device_type, idx):
    """Set current device for the given device type."""
    if device_type == 'npu':
        torch.npu.set_device(idx)
    else:
        torch.cuda.set_device(idx)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _load_config(model_id: str, local_files_only: bool):
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)


def _ref_path(port: int) -> str:
    return f'/tmp/twinkle_ep_fsdp_ckpt_ref_{port}.pt'


def _ckpt_dir(port: int) -> str:
    return f'/tmp/twinkle_ep_fsdp_ckpt_save_{port}'


def _load_model(model_id, local_only, device, num_layers=1):
    config = _load_config(model_id, local_only)
    if hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = num_layers
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    if hasattr(config, '_experts_implementation'):
        config._experts_implementation = 'eager'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_only)
    model.to(device)
    return model


def _collect_ep_expert_names(model):
    """Collect FQNs of expert parameters from EP-patched modules.

    Matches against model.named_parameters() output by building the same
    FQN prefix that named_parameters() would produce, then cross-checking
    against the actual parameter names to avoid silent mismatches after
    FSDP wrapping.
    """
    # First pass: build candidate names from module tree
    candidate_names = set()
    for fqn, module in model.named_modules():
        if not getattr(module, '_ep_patched', False):
            continue
        experts = getattr(module, 'experts', None)
        if experts is None:
            continue
        experts_prefix = fqn + '.experts.' if fqn else 'experts.'
        for pname, _ in experts.named_parameters():
            candidate_names.add(experts_prefix + pname)

    # Second pass: cross-check against actual named_parameters() to catch
    # any FSDP-induced name changes (e.g. _fsdp_wrapped_module. prefixes)
    actual_param_names = {n for n, _ in model.named_parameters()}
    matched = candidate_names & actual_param_names
    unmatched = candidate_names - actual_param_names
    if unmatched:
        print(f'[WARNING] _collect_ep_expert_names: {len(unmatched)} candidate '
              f'expert param names not found in model.named_parameters() — '
              f'they will NOT be all-gathered. Examples: {list(unmatched)[:3]}')
    return matched


def _gather_full_state_dict(model, ep_fsdp_mesh):
    """Collect full state dict with EP-aware all-gather for expert params.

    For non-expert params, uses to_local_tensor() (DTensor.full_tensor()).
    For expert params, additionally all-gathers across the EP group on dim-0
    to reconstruct the full [num_experts, ...] tensor.

    Raises RuntimeError if EP is expected but the EP group cannot be obtained,
    to prevent silent fallback to saving only local expert shards.
    """
    ep_group = None
    ep_ws = 1
    if ep_fsdp_mesh is not None:
        # Let errors propagate — a silent fallback here would save only local
        # expert shards and produce a checkpoint that looks correct but isn't.
        ep_group = ep_fsdp_mesh['ep'].get_group()
        ep_ws = ep_fsdp_mesh['ep'].size()

    ep_expert_names = set()
    if ep_ws > 1:
        ep_expert_names = _collect_ep_expert_names(model)
        if not ep_expert_names:
            raise RuntimeError(
                '_gather_full_state_dict: EP is enabled (ep_ws=%d) but no expert '
                'parameter names were found via _collect_ep_expert_names(). '
                'Expert params will NOT be all-gathered — checkpoint would be wrong. '
                'Check that apply_expert_parallel() was called before wrap_model() '
                'and that _ep_patched is set on MoE blocks.' % ep_ws)
        print(f'[gather_full_state_dict] EP all-gather enabled: ep_ws={ep_ws}, '
              f'expert params to gather={len(ep_expert_names)}')

    gathered_count = 0
    direct_count = 0
    full_sd = {}
    for name, param in model.named_parameters():
        local_full = torch_util.to_local_tensor(param)
        if name in ep_expert_names and ep_ws > 1 and ep_group is not None:
            local_full = local_full.contiguous().to(param.device)
            gathered = [torch.empty_like(local_full) for _ in range(ep_ws)]
            dist.all_gather(gathered, local_full, group=ep_group)
            local_full = torch.cat(gathered, dim=0)
            # Sanity-check: dim-0 of the gathered tensor must equal num_experts,
            # not num_experts // ep_size (which would indicate a missing all-gather).
            expected_full_dim0 = local_full.shape[0]
            shard_dim0 = gathered[0].shape[0]
            assert expected_full_dim0 == shard_dim0 * ep_ws, (
                f'Expert param {name}: gathered dim-0={expected_full_dim0} != '
                f'shard_dim0={shard_dim0} * ep_ws={ep_ws}. All-gather may be wrong.')
            gathered_count += 1
        else:
            direct_count += 1
        full_sd[name] = local_full.cpu()

    print(f'[gather_full_state_dict] params via EP all-gather={gathered_count}, '
          f'params via direct={direct_count}, total={len(full_sd)}')
    return full_sd


def _clean_name(name: str) -> str:
    """Strip FSDP1 wrapper prefixes from parameter names.

    Note: FSDP2 (fully_shard) preserves original parameter names and does NOT
    add these prefixes. This function is kept for compatibility but is effectively
    a no-op for FSDP2-wrapped models.
    """
    name = name.replace('_fsdp_wrapped_module.', '')
    name = name.replace('module.', '')
    return name


def _run_save_reference(rank, world_size, port, model_id, local_only):
    """Save full state dict from single-device model as reference."""
    os.environ.update({
        'RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_RANK': '0',
        'LOCAL_WORLD_SIZE': '1',
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
    })
    device_type, _, _ = _get_device_info()
    device = torch.device(f'{device_type}:0')
    _set_device(device_type, 0)

    model = _load_model(model_id, local_only, device)
    ref_sd = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(ref_sd, _ref_path(port))
    print(f'[Reference] Saved {len(ref_sd)} parameters')


def _run_ep_fsdp_save(rank, world_size, port, model_id, local_only):
    """Load model with EP+FSDP, collect full state dict via EP-aware all-gather."""
    os.environ.update({
        'RANK': str(rank),
        'WORLD_SIZE': str(world_size),
        'LOCAL_RANK': str(rank),
        'LOCAL_WORLD_SIZE': str(world_size),
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
    })
    device_type, backend, _ = _get_device_info()
    device = torch.device(f'{device_type}:{rank}')
    _set_device(device_type, rank)

    if device_type == 'npu':
        from twinkle.utils.platforms.npu import ensure_hccl_socket_env
        ensure_hccl_socket_env(port)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}',
        device_id=device,
        timeout=timedelta(minutes=10))
    dist.barrier()

    try:
        ep_size = 2
        device_mesh = DeviceMesh(
            device_type=device_type,
            mesh=np.arange(world_size).reshape(world_size),
            mesh_dim_names=('fsdp',),
            ep_size=ep_size,
        )

        model = _load_model(model_id, local_only, device)

        # Build strategy — it creates ep_fsdp_device_mesh internally from ep_size
        fsdp = NativeFSDPStrategy(
            device_mesh=device_mesh,
            mixed_precision='no',
            fsdp_config={},
            enable_ep=True,
            ep_size=ep_size,
        )
        ep_fsdp_mesh = fsdp.ep_fsdp_device_mesh

        # Apply EP before FSDP wrapping
        apply_expert_parallel(
            getattr(model, 'model', model),
            device_mesh,
            config={
                'enabled': True,
                'router_dtype': 'fp32',
                'keep_router_logits': False,
            },
            ep_fsdp_device_mesh=ep_fsdp_mesh,
        )

        model, _ = fsdp.wrap_model(model, optimizer=None)

        # Collect full state dict with EP-aware all-gather
        full_sd = _gather_full_state_dict(model, ep_fsdp_mesh)

        # Save from rank 0 only
        ckpt_dir = _ckpt_dir(port)
        if rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            cleaned_sd = {_clean_name(k): v for k, v in full_sd.items()}
            torch.save(cleaned_sd, os.path.join(ckpt_dir, 'state_dict.pt'))
            print(f'[EP+FSDP rank {rank}] Saved {len(cleaned_sd)} params, '
                  f'EP expert names: {len(_collect_ep_expert_names(model))}')

        dist.barrier()
    except Exception as e:
        print(f'Rank {rank} error: {e}')
        raise
    finally:
        dist.destroy_process_group()


class TestEPFSDPCheckpointSave(unittest.TestCase):

    def test_checkpoint_round_trip(self):
        """Verify EP+FSDP save produces state dict matching single-device reference.

        This test catches the bug where only local expert shards (num_experts/ep_size)
        were saved instead of the full expert tensor (num_experts).
        """
        if not dist.is_available():
            self.skipTest('Need distributed')

        device_type, _, device_count = _get_device_info()
        if device_count < 4:
            self.skipTest(f'Need 4 {device_type} devices, found {device_count}')

        model_id = os.environ.get('QWEN3_MOE_MODEL_ID', 'Qwen/Qwen3-30B-A3B-Instruct-2507')
        local_only = os.environ.get('QWEN3_MOE_LOCAL_ONLY', '1') != '0'

        try:
            _load_config(model_id, local_only)
        except Exception as e:
            self.skipTest(f'Model not available: {e}')

        port = _find_free_port()
        port2 = _find_free_port()
        ref_path = _ref_path(port)

        try:
            # 1. Save single-GPU reference state dict
            mp.spawn(_run_save_reference,
                     args=(1, port, model_id, local_only),
                     nprocs=1, join=True)

            # 2. Save EP+FSDP checkpoint (4 GPUs, ep_size=2)
            mp.spawn(_run_ep_fsdp_save,
                     args=(4, port2, model_id, local_only),
                     nprocs=4, join=True)

            # 3. Compare saved checkpoint against single-GPU reference
            ref_sd = torch.load(ref_path, weights_only=True)
            saved_sd = torch.load(
                os.path.join(_ckpt_dir(port2), 'state_dict.pt'),
                weights_only=True)

            # Every reference param must exist in saved checkpoint
            missing = set(ref_sd.keys()) - set(saved_sd.keys())
            self.assertEqual(
                len(missing), 0,
                f'Missing params in saved checkpoint: {missing}')

            # Saved checkpoint must not contain extra params not in reference
            extra = set(saved_sd.keys()) - set(ref_sd.keys())
            self.assertEqual(
                len(extra), 0,
                f'Extra params in saved checkpoint (not in reference): {extra}')

            # Shapes and values must match
            mismatched_shape = []
            mismatched_value = []
            for name in ref_sd:
                ref_t = ref_sd[name]
                saved_t = saved_sd[name]
                if ref_t.shape != saved_t.shape:
                    mismatched_shape.append(
                        f'{name}: ref={ref_t.shape} saved={saved_t.shape}')
                    continue
                if not torch.allclose(ref_t.float(), saved_t.float(),
                                      rtol=REL_TOL, atol=ABS_TOL):
                    diff = (ref_t.float() - saved_t.float()).abs().max().item()
                    mismatched_value.append(f'{name}: max_diff={diff:.2e}')

            self.assertEqual(
                len(mismatched_shape), 0,
                'Shape mismatches:\n' + '\n'.join(mismatched_shape))
            self.assertEqual(
                len(mismatched_value), 0,
                'Value mismatches:\n' + '\n'.join(mismatched_value))

            print(f'[PASS] All {len(ref_sd)} parameters match between '
                  f'single-device reference and EP+FSDP checkpoint')

        finally:
            if os.path.exists(ref_path):
                os.remove(ref_path)
            for d in [_ckpt_dir(port), _ckpt_dir(port2)]:
                if os.path.exists(d):
                    shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()