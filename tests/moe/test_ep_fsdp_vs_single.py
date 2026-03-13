# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test EP+FSDP vs single-GPU precision:
1. Forward Logits & Loss
2. Gradients (non-expert and expert layers)
3. Updated Weights after optimizer step

Requirements:
  - 4 CUDA GPUs
  - Model weights accessible via QWEN3_MOE_MODEL_ID (default: Qwen/Qwen3-30B-A3B-Instruct-2507)

Launch (requires 4 CUDA GPUs; skipped automatically if fewer GPUs are available):

    pytest tests/moe/test_ep_fsdp_vs_single.py -v -s

    # To use a local model:
    QWEN3_MOE_MODEL_ID=/path/to/model pytest tests/moe/test_ep_fsdp_vs_single.py -v -s

Note: nproc_per_node=1 is intentional — the test internally spawns 4 worker processes via
mp.spawn (1 for the single-GPU baseline and 4 for the EP+FSDP run).
"""

import numpy as np
import os
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

ABS_TOL = 5e-3
LOSS_TOL = 1e-4
REL_TOL = 1e-4


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _load_config(model_id: str, local_files_only: bool):
    return AutoConfig.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)


def _single_snapshot_path(port: int) -> str:
    return f'/tmp/twinkle_ep_fsdp_vs_single_{port}.pt'


def _load_model(model_id: str, local_only: bool, device: torch.device, num_layers: int = 1):
    config = _load_config(model_id, local_only)
    if hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = num_layers
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    if hasattr(config, '_experts_implementation'):
        config._experts_implementation = 'eager'

    # Disable dropout to ensure determinism
    dropout_attrs = ['attention_dropout', 'hidden_dropout', 'classifier_dropout', 'resid_pdrop', 'embd_pdrop']
    for attr in dropout_attrs:
        if hasattr(config, attr):
            setattr(config, attr, 0.0)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_only)
    model.to(device)
    return model


def _clean_name(name: str) -> str:
    """Strip FSDP wrapper prefixes from parameter names."""
    name = name.replace('_fsdp_wrapped_module.', '')
    name = name.replace('module.', '')
    return name


def _get_full_tensor(tensor_obj):
    """Reconstruct a DTensor to a plain CPU tensor; pass through plain tensors unchanged."""
    if tensor_obj is None:
        return None
    if hasattr(tensor_obj, 'full_tensor'):
        return tensor_obj.full_tensor().detach().cpu()
    elif hasattr(tensor_obj, '_local_tensor'):
        return tensor_obj._local_tensor.detach().cpu()
    else:
        return tensor_obj.detach().cpu()


def _split_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    if world_size <= 1:
        return 0, total
    if rank < 0 or rank >= world_size:
        return 0, 0
    base, rem = divmod(total, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def _match_single_tensor_for_compare(mapped_name: str, multi_tensor: torch.Tensor, single_dict: dict, ep_rank: int,
                                     fsdp_rank: int, fsdp_world_size: int):
    """Return the single-GPU tensor sliced to match multi_tensor; expert params support dim0(ep)+dim1(fsdp) sharding."""
    single_tensor = single_dict.get(mapped_name)
    if single_tensor is None:
        return None

    if multi_tensor.shape == single_tensor.shape:
        return single_tensor

    # EP/FSDP expert parameter sharding rules:
    # - dim0 sharded by EP: single=[num_experts,...] -> local experts
    # - dim1 sharded by FSDP: local experts slice then cut dim1
    if 'experts.' not in mapped_name:
        return None
    if multi_tensor.ndim < 1 or single_tensor.ndim < 1:
        return None
    candidate = single_tensor

    # 1) dim0: ep shard
    if candidate.ndim >= 1 and candidate.shape[0] != multi_tensor.shape[0]:
        local_experts = multi_tensor.shape[0]
        total_experts = candidate.shape[0]
        if local_experts == 0 or total_experts % local_experts != 0:
            return None
        ep_world_size = total_experts // local_experts
        if ep_rank is None or ep_rank < 0 or ep_rank >= ep_world_size:
            return None
        start0 = ep_rank * local_experts
        end0 = start0 + local_experts
        candidate = candidate[start0:end0]

    # 2) dim1: fsdp shard
    if candidate.ndim >= 2 and candidate.shape[1] != multi_tensor.shape[1]:
        if fsdp_world_size is None or fsdp_world_size <= 1:
            return None
        if fsdp_rank is None:
            return None
        start1, end1 = _split_range(candidate.shape[1], int(fsdp_rank), int(fsdp_world_size))
        if end1 <= start1:
            return None
        candidate = candidate[:, start1:end1, ...]

    if candidate.shape != multi_tensor.shape:
        return None
    return candidate


def _run_single_gpu(rank, world_size, port, model_id, local_only):
    """Single GPU baseline."""
    os.environ.update({
        'RANK': '0',
        'WORLD_SIZE': '1',
        'LOCAL_RANK': '0',
        'LOCAL_WORLD_SIZE': '1',
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port)
    })

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.manual_seed(1234)

    model = _load_model(model_id, local_only, device, num_layers=1)
    model.train()
    vocab_size = int(model.config.vocab_size)

    batch_size, seq_len = 2, 1024
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    labels = input_ids.clone()
    labels[:, 0] = -100

    outputs = model(input_ids=input_ids, position_ids=position_ids, labels=labels, use_cache=False)
    loss = outputs.loss
    loss.backward()

    # Collect grads and weights
    grad_dict = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.grad is not None}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    opt.step()
    new_weight_dict = {n: p.detach().cpu() for n, p in model.named_parameters()}

    # Save inputs so multi-GPU run uses identical data
    torch.save(
        {
            'input_ids': input_ids.cpu(),
            'position_ids': position_ids.cpu(),
            'logits': outputs.logits.detach().cpu(),
            'loss': loss.item(),
            'grad_dict': grad_dict,
            'new_weight_dict': new_weight_dict,
        }, _single_snapshot_path(port))

    print(f'[Single] Loss={loss.item():.4f}')


def _run_multi_gpu(rank, world_size, port, model_id, local_only):
    """4-GPU EP+FSDP with two independent meshes."""
    os.environ.update({
        'RANK': str(rank),
        'WORLD_SIZE': str(world_size),
        'LOCAL_RANK': str(rank),
        'LOCAL_WORLD_SIZE': str(world_size),
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port)
    })

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://127.0.0.1:{port}',
        device_id=device,
        timeout=timedelta(minutes=15))
    dist.barrier()

    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        # New design: main mesh does NOT include 'ep' dimension
        # Main mesh: fsdp=4 (all 4 GPUs for FSDP)
        # ep_size=2 is stored as attribute, used to build separate ep_fsdp_device_mesh
        ep_size = 2
        device_mesh = DeviceMesh(
            device_type='cuda',
            mesh=np.arange(world_size).reshape(world_size),  # 1D mesh: [0, 1, 2, 3]
            mesh_dim_names=('fsdp', ),
            ep_size=ep_size,  # ep_size as attribute, not mesh dimension
        )

        model = _load_model(model_id, local_only, device, num_layers=1)
        model.train()

        # Load inputs saved by the single-GPU run
        single_data = torch.load(_single_snapshot_path(port), weights_only=True)
        input_ids = single_data['input_ids'].to(device)
        position_ids = single_data['position_ids'].to(device)
        labels = input_ids.clone()
        labels[:, 0] = -100

        # Build explicit ep_fsdp_device_mesh
        ep_fsdp_mesh = device_mesh.build_ep_fsdp_device_mesh(ep_size=ep_size)

        # Apply EP with ep_fsdp_device_mesh
        apply_expert_parallel(
            getattr(model, 'model', model),
            device_mesh,
            config={
                'enabled': True,
                'router_dtype': 'fp32',
                'keep_router_logits': False
            },
            ep_fsdp_device_mesh=ep_fsdp_mesh,
        )

        # FSDP2 wrap
        fsdp = NativeFSDPStrategy(
            device_mesh=device_mesh,
            mixed_precision='no',
            fsdp_config={},
            enable_ep=True,
            ep_fsdp_device_mesh=ep_fsdp_mesh,
        )
        model, _ = fsdp.wrap_model(model, optimizer=None)

        outputs = model(input_ids=input_ids, position_ids=position_ids, labels=labels, use_cache=False)
        loss = outputs.loss
        loss.backward()

        # Collect gradients — reconstruct DTensors via full_tensor()
        grad_dict = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_dict[n] = _get_full_tensor(p.grad)

        # Optimizer step
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
        opt.step()

        # Weights after optimizer step
        new_weight_dict = {}
        for n, p in model.named_parameters():
            new_weight_dict[n] = _get_full_tensor(p)

        # Compare — all ranks participate
        single_grads = single_data['grad_dict']
        single_weights = single_data['new_weight_dict']

        # Get EP rank from ep_fsdp_mesh
        ep_rank = ep_fsdp_mesh.get_local_rank('ep') if ep_fsdp_mesh is not None else 0
        fsdp_rank = device_mesh.fsdp_rank or 0
        fsdp_world_size = device_mesh.fsdp_world_size or 1

        # Forward comparison is computed only on rank 0, then broadcast to avoid other ranks hanging at barrier
        forward_err = None
        if rank == 0:
            single_logits = single_data['logits']
            multi_logits = _get_full_tensor(outputs.logits)
            logits_abs_diff = (single_logits - multi_logits).abs()
            logits_max_diff = logits_abs_diff.max().item()
            logits_mean_diff = logits_abs_diff.mean().item()
            print('\n=== Forward: Logits ===')
            print(f'  Max diff: {logits_max_diff:.2e}, Mean diff: {logits_mean_diff:.2e}')
            if not torch.allclose(single_logits, multi_logits, rtol=REL_TOL, atol=ABS_TOL):
                forward_err = f'Logits mismatch! Max diff: {logits_max_diff}, Mean diff: {logits_mean_diff}'

            print('\n=== Forward: Loss ===')
            print(f'  Single: {single_data["loss"]:.6f}, Multi: {loss.item():.6f}')
            loss_diff = abs(single_data['loss'] - loss.item())
            single_loss_t = torch.tensor(single_data['loss'], dtype=torch.float32)
            multi_loss_t = torch.tensor(loss.item(), dtype=torch.float32)
            if (not torch.allclose(single_loss_t, multi_loss_t, rtol=REL_TOL, atol=LOSS_TOL) and forward_err is None):
                forward_err = f'Loss mismatch! Diff: {loss_diff}'

        obj = [forward_err]
        dist.broadcast_object_list(obj, src=0)
        if obj[0] is not None:
            raise AssertionError(obj[0])

        # Compare non-expert gradients
        print(f'\n=== Rank {rank}: Gradients (non-expert) ===')
        verified = 0
        seen_mapped = set()
        for n in grad_dict:
            mapped_n = _clean_name(n)
            if mapped_n in seen_mapped or 'experts.' in mapped_n:
                continue
            seen_mapped.add(mapped_n)
            m_grad = grad_dict[n]
            s_grad = _match_single_tensor_for_compare(
                mapped_n,
                m_grad,
                single_grads,
                ep_rank,
                fsdp_rank,
                fsdp_world_size,
            )
            if s_grad is None:
                continue
            grad_abs_diff = (s_grad - m_grad).abs()
            grad_max_diff = grad_abs_diff.max().item()
            grad_mean_diff = grad_abs_diff.mean().item()
            is_close = torch.allclose(s_grad, m_grad, rtol=REL_TOL, atol=ABS_TOL)
            status = 'PASS' if is_close else 'FAIL'
            print(f'  [{status}] {mapped_n}: max_diff={grad_max_diff:.2e}, mean_diff={grad_mean_diff:.2e}')
            assert is_close, f'Grad mismatch for {mapped_n}! Max diff: {grad_max_diff}, Mean diff: {grad_mean_diff}'
            verified += 1
        assert verified > 0, f'Error: No non-expert gradients were verified on rank {rank}!'

        # Compare expert gradients
        print(f'\n=== Rank {rank}: Gradients (expert) ===')
        verified = 0
        ratio_list = []
        seen_mapped = set()
        for n in grad_dict:
            mapped_n = _clean_name(n)
            if mapped_n in seen_mapped or 'experts.' not in mapped_n:
                continue
            seen_mapped.add(mapped_n)
            m_grad = grad_dict[n]
            s_grad = _match_single_tensor_for_compare(
                mapped_n,
                m_grad,
                single_grads,
                ep_rank,
                fsdp_rank,
                fsdp_world_size,
            )
            if s_grad is None:
                continue
            grad_abs_diff = (s_grad - m_grad).abs()
            grad_max_diff = grad_abs_diff.max().item()
            grad_mean_diff = grad_abs_diff.mean().item()
            s_norm = s_grad.float().norm().item()
            m_norm = m_grad.float().norm().item()
            ratio = m_norm / (s_norm + 1e-12)
            ratio_list.append(ratio)
            is_close = torch.allclose(s_grad, m_grad, rtol=REL_TOL, atol=ABS_TOL)
            status = 'PASS' if is_close else 'FAIL'
            print(f'  [{status}] {mapped_n}: max_diff={grad_max_diff:.2e}, mean_diff={grad_mean_diff:.2e}, '
                  f'norm_ratio(ep/single)={ratio:.4f}')
            assert is_close, (f'Expert grad mismatch for {mapped_n}! '
                              f'Max diff: {grad_max_diff}, Mean diff: {grad_mean_diff}, Ratio: {ratio}')
            verified += 1
        if verified == 0:
            print(f'  [INFO] No expert gradients matched on rank {rank} (EP distribution is expected)')
        else:
            ratio_t = torch.tensor(ratio_list, dtype=torch.float32)
            print(f'  [INFO] expert grad norm ratio(ep/single): '
                  f'min={ratio_t.min().item():.4f}, max={ratio_t.max().item():.4f}, mean={ratio_t.mean().item():.4f}')

        # Compare updated weights
        print(f'\n=== Rank {rank}: Updated Weights ===')
        verified = 0
        seen_mapped = set()
        for n in new_weight_dict:
            mapped_n = _clean_name(n)
            if mapped_n in seen_mapped:
                continue
            seen_mapped.add(mapped_n)
            m_w = new_weight_dict[n]
            s_w = _match_single_tensor_for_compare(
                mapped_n,
                m_w,
                single_weights,
                ep_rank,
                fsdp_rank,
                fsdp_world_size,
            )
            if s_w is None:
                continue
            weight_abs_diff = (s_w - m_w).abs()
            weight_max_diff = weight_abs_diff.max().item()
            weight_mean_diff = weight_abs_diff.mean().item()
            is_close = torch.allclose(s_w, m_w, rtol=REL_TOL, atol=ABS_TOL)
            status = 'PASS' if is_close else 'FAIL'
            print(f'  [{status}] {mapped_n}: max_diff={weight_max_diff:.2e}, mean_diff={weight_mean_diff:.2e}')
            assert is_close, (
                f'Weight mismatch for {mapped_n}! Max diff: {weight_max_diff}, Mean diff: {weight_mean_diff}')
            verified += 1
        assert verified > 0, f'Error: No weights were verified on rank {rank}!'

        dist.barrier()
    except Exception as e:
        print(f'Rank {rank} error: {e}')
        raise
    finally:
        dist.destroy_process_group()


class TestEPFSDPvsSingle(unittest.TestCase):

    def test_alignment(self):
        if not dist.is_available() or not torch.cuda.is_available():
            self.skipTest('Need distributed + CUDA')
        if torch.cuda.device_count() < 4:
            self.skipTest('Need 4 GPUs')

        model_id = os.environ.get('QWEN3_MOE_MODEL_ID', 'Qwen/Qwen3-30B-A3B-Instruct-2507')
        local_only = os.environ.get('QWEN3_MOE_LOCAL_ONLY', '1') != '0'

        try:
            _load_config(model_id, local_only)
        except Exception as e:
            self.skipTest(f'Model not available: {e}')

        port = _find_free_port()
        snapshot_path = _single_snapshot_path(port)

        try:
            # Run single GPU baseline
            mp.spawn(_run_single_gpu, args=(1, port, model_id, local_only), nprocs=1, join=True)

            # Run 4-GPU EP+FSDP
            mp.spawn(_run_multi_gpu, args=(4, port, model_id, local_only), nprocs=4, join=True)
        finally:
            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)


if __name__ == '__main__':
    unittest.main()
