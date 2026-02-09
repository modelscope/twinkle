# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import socket
import sys
import contextlib
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from twinkle.model.transformers.strategy.sequence_parallel import (
    DistributedAttention,
    _get_sp_group_from_device_mesh,
    sequence_parallel,
)
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.utils import DeviceMesh


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _broadcast_params(module: torch.nn.Module) -> None:
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def _enable_strict_determinism() -> None:
    # Reduce kernel variability: avoid TF32 and prefer deterministic algorithms.
    # Note: some collectives/kernels may not have deterministic variants; keep warn_only.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)

def _to_local(x: torch.Tensor) -> torch.Tensor:
    # FSDP2 grads can be DTensors; unwrap to local for comparison.
    if hasattr(x, "to_local"):
        try:
            return x.to_local()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(x, "local_tensor"):
        try:
            return x.local_tensor
        except Exception:  # noqa: BLE001
            pass
    return x


@contextlib.contextmanager
def _force_sdpa_math():
    """Force SDPA to use the math backend for stricter (more deterministic) alignment."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except Exception:  # noqa: BLE001
        pass

    # Fallback for older torch versions.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        ):
            yield
    else:
        yield


class _SingleAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, sp_enabled: bool):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sp_enabled = sp_enabled

        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self._dist_attn = DistributedAttention(self._local_attn, sequence_parallel)

    @staticmethod
    def _local_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        position_ids=None,
        is_causal: bool = True,
        **_kwargs,
    ) -> torch.Tensor:
        # query/key/value: [B, S, H, D]
        q = query.permute(0, 2, 1, 3).contiguous()
        k = key.permute(0, 2, 1, 3).contiguous()
        v = value.permute(0, 2, 1, 3).contiguous()
        with _force_sdpa_math():
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        return out.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)

        if self.sp_enabled and sequence_parallel.world_size and sequence_parallel.world_size > 1:
            ctx = self._dist_attn(q, k, v, None, position_ids=None, is_causal=True)
        else:
            ctx = self._local_attn(q, k, v, None, position_ids=None, is_causal=True)

        out = self.out_proj(ctx.reshape(bsz, seqlen, self.hidden_dim))
        return out


def _init_dist(rank: int, world_size: int, port: int) -> torch.device:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
        timeout=timedelta(minutes=15),
    )
    dist.barrier()
    return device


def _setup_sp(device_mesh: DeviceMesh, sp_size: int) -> None:
    sequence_parallel.world_size = sp_size
    sequence_parallel._init_device_mesh(device_mesh)


def _run_worker_single_attn(rank: int, world_size: int, port: int, padding: bool):
    device = _init_dist(rank, world_size, port)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        _enable_strict_determinism()

        # Align with VeOmni's test semantics: one SP group across the whole world.
        sp_size = world_size
        device_mesh = DeviceMesh.from_sizes(dp_size=world_size, ulysses_size=sp_size, device_type="cuda")
        _setup_sp(device_mesh, sp_size)
        sp_group = _get_sp_group_from_device_mesh(device_mesh, sp_size)

        batch_size = 2
        unpad_seq_len = 127 if padding else 128
        hidden_dim = 256
        num_heads = 16  # must be divisible by sp_size
        assert num_heads % sp_size == 0
        dtype = torch.float32  # maximize determinism for strict alignment

        full_x = torch.randn(batch_size, unpad_seq_len, hidden_dim, device=device, dtype=dtype)
        dist.broadcast(full_x, src=0)

        if padding:
            x = sequence_parallel.pad(full_x, padding_value=0.0, position_ids=None, dim=1)
        else:
            x = full_x
        pad_seq_len = x.size(1)
        assert pad_seq_len % sp_size == 0

        dp_x = x.detach().requires_grad_(True)
        sp_x_local = sequence_parallel.split(x, dim=1, position_ids=None).detach().requires_grad_(True)
        sp_rank = dist.get_rank(sp_group) if sp_group is not None else 0
        local = pad_seq_len // sp_size
        start = sp_rank * local
        end = start + local
        valid_end = min(end, unpad_seq_len)
        local_valid = max(valid_end - start, 0)

        attn_sp = _SingleAttention(hidden_dim, num_heads, sp_enabled=True).to(device=device, dtype=dtype)
        attn_dp = _SingleAttention(hidden_dim, num_heads, sp_enabled=False).to(device=device, dtype=dtype)
        _broadcast_params(attn_sp)
        attn_dp.load_state_dict(attn_sp.state_dict())

        # forward (SP)
        sp_out_local = attn_sp(sp_x_local)
        sp_out_full = sequence_parallel.gather(sp_out_local, dim=1, position_ids=None)[:, :unpad_seq_len]

        # backward (SP): VeOmni uses overlapping grad on local output, then all-reduces param grads.
        sp_loss = sp_out_local[:, :local_valid].sum() * 2.0
        sp_loss.backward()
        sp_q_grad = attn_sp.q_proj.weight.grad.detach().clone()
        sp_o_grad = attn_sp.out_proj.weight.grad.detach().clone()
        sp_x_grad_full = sequence_parallel.gather(sp_x_local.grad.detach(), dim=1, position_ids=None)[:, :unpad_seq_len]

        if sp_group is not None:
            dist.all_reduce(sp_q_grad, op=dist.ReduceOp.SUM, group=sp_group)
            dist.all_reduce(sp_o_grad, op=dist.ReduceOp.SUM, group=sp_group)

        # Disable SP for the baseline DP run (like set_ulysses_sequence_parallel_group(None)).
        saved_world = sequence_parallel.world_size
        saved_sp_world = sequence_parallel.sp_world_size
        saved_group = sequence_parallel._sp_group
        sequence_parallel.world_size = 1
        sequence_parallel.sp_world_size = 1
        sequence_parallel._sp_group = None

        # forward/backward (DP full sequence)
        dp_out_full = attn_dp(dp_x)[:, :unpad_seq_len]
        torch.testing.assert_close(dp_out_full, sp_out_full, atol=1e-6, rtol=1e-5)

        dp_loss = dp_out_full.sum() * 2.0
        dp_loss.backward()
        dp_q_grad = attn_dp.q_proj.weight.grad.detach()
        dp_o_grad = attn_dp.out_proj.weight.grad.detach()
        dp_x_grad_full = dp_x.grad.detach()[:, :unpad_seq_len]

        # Restore SP globals (not strictly needed, but keeps teardown clean).
        sequence_parallel.world_size = saved_world
        sequence_parallel.sp_world_size = saved_sp_world
        sequence_parallel._sp_group = saved_group

        torch.testing.assert_close(dp_o_grad, sp_o_grad, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(dp_q_grad, sp_q_grad, atol=1e-3, rtol=1e-4)
        torch.testing.assert_close(dp_x_grad_full, sp_x_grad_full, atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def _run_worker_single_attn_fsdp(rank: int, world_size: int, port: int):
    device = _init_dist(rank, world_size, port)
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        _enable_strict_determinism()

        # Use sp_size=world_size to avoid multiple independent SP groups while FSDP shards globally.
        sp_size = world_size
        # For FSDP+SP, SP is derived from dp/fsdp ranks. Use fsdp=world, dp=1.
        device_mesh = DeviceMesh.from_sizes(fsdp_size=world_size, dp_size=1, ulysses_size=sp_size, device_type="cuda")
        _setup_sp(device_mesh, sp_size)
        sp_group = _get_sp_group_from_device_mesh(device_mesh, sp_size)

        batch_size = 2
        unpad_seq_len = 128
        hidden_dim = 256
        num_heads = 16
        assert num_heads % sp_size == 0
        dtype = torch.float32

        full_x = torch.randn(batch_size, unpad_seq_len, hidden_dim, device=device, dtype=dtype)
        dist.broadcast(full_x, src=0)

        dp_x = full_x.detach().requires_grad_(True)
        sp_x_local = sequence_parallel.split(full_x, dim=1, position_ids=None).detach().requires_grad_(True)

        attn_sp = _SingleAttention(hidden_dim, num_heads, sp_enabled=True).to(device=device, dtype=dtype)
        attn_dp = _SingleAttention(hidden_dim, num_heads, sp_enabled=False).to(device=device, dtype=dtype)
        _broadcast_params(attn_sp)
        attn_dp.load_state_dict(attn_sp.state_dict())

        fsdp = NativeFSDPStrategy(device_mesh=device_mesh, mixed_precision="no", fsdp_config={})
        attn_sp, _ = fsdp.wrap_model(attn_sp, optimizer=None)
        attn_dp, _ = fsdp.wrap_model(attn_dp, optimizer=None)

        # SP forward/backward: local loss; across ranks sum(loss_local) == DP full loss.
        sp_out_local = attn_sp(sp_x_local)
        sp_out_full = sequence_parallel.gather(sp_out_local, dim=1, position_ids=None)[:, :unpad_seq_len]
        sp_loss = sp_out_local.sum() * 2.0
        sp_loss.backward()
        sp_x_grad_local = sp_x_local.grad.detach()
        sp_x_grad_full = sequence_parallel.gather(sp_x_grad_local, dim=1, position_ids=None)[:, :unpad_seq_len]

        # DP forward/backward: full loss on full sequence.
        dp_out_full = attn_dp(dp_x)[:, :unpad_seq_len]
        torch.testing.assert_close(dp_out_full, sp_out_full, atol=1e-6, rtol=1e-5)
        # In FSDP, per-rank losses are effectively summed across ranks when forming param grads.
        # DP uses identical full-seq inputs on every rank, so scale by world_size to match the
        # global objective of SP (which partitions the sequence across ranks).
        dp_loss = dp_out_full.sum() * (2.0 / float(world_size))
        dp_loss.backward()

        # Under FSDP2, grads are sharded; compare local shards directly (same mesh, same wrapping).
        torch.testing.assert_close(
            _to_local(attn_dp.out_proj.weight.grad.detach()),
            _to_local(attn_sp.out_proj.weight.grad.detach()),
            atol=1e-3,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            _to_local(attn_dp.q_proj.weight.grad.detach()),
            _to_local(attn_sp.q_proj.weight.grad.detach()),
            atol=1e-3,
            rtol=1e-4,
        )
        # dp_x.grad is not reduced across ranks; rescale to match the unscaled SP full loss.
        dp_x_grad_full = dp_x.grad.detach()[:, :unpad_seq_len] * float(world_size)
        torch.testing.assert_close(dp_x_grad_full, sp_x_grad_full, atol=1e-5, rtol=1e-5)

        # Only validate forward + parameter grads for the FSDP+SP case.
    finally:
        dist.destroy_process_group()


class TestSequenceParallelSingleAttention(unittest.TestCase):
    def test_single_attention(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for sequence-parallel attention test.")
        port = _find_free_port()
        mp.spawn(
            _run_worker_single_attn,
            args=(world_size, port, False),
            nprocs=world_size,
            join=True,
        )

    def test_single_attention_padding(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for sequence-parallel attention test.")
        port = _find_free_port()
        mp.spawn(
            _run_worker_single_attn,
            args=(world_size, port, True),
            nprocs=world_size,
            join=True,
        )

    def test_single_attention_fsdp(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 4
        if torch.cuda.device_count() < world_size:
            self.skipTest("Requires at least 4 GPUs for sequence-parallel attention test.")
        port = _find_free_port()
        mp.spawn(
            _run_worker_single_attn_fsdp,
            args=(world_size, port),
            nprocs=world_size,
            join=True,
        )
