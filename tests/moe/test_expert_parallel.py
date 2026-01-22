# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import socket
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import copy
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from twinkle.model.moe import apply_expert_parallel
from twinkle.utils import DeviceMesh


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Qwen3MoeLikeBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, norm_topk_prob: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_experts)])

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[int(expert_idx)]
            idx, top_x = torch.where(expert_mask[int(expert_idx)].squeeze(0))
            token_idx = top_x
            topk_pos = idx
            current_state = hidden_states[token_idx]
            current_hidden_states = expert_layer(current_state) * routing_weights[token_idx, topk_pos, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states, router_logits


def _run_worker(rank: int, world_size: int, port: int):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA (2 GPUs).")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        device_id=device,
    )
    dist.barrier()

    try:
        torch.manual_seed(1234)
        model = Qwen3MoeLikeBlock(hidden_size=16, num_experts=4, top_k=2).to(device)
        baseline = copy.deepcopy(model)

        inputs = torch.randn(2, 4, 16, device=device)
        baseline_out, baseline_logits = baseline(inputs)

        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(world_size),
            mesh_dim_names=("ep",),
        )
        apply_expert_parallel(
            model,
            device_mesh,
            config={
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": True,
            },
        )
        dist.barrier()
        assert hasattr(model, "_ep_local_start")
        assert len(model.experts) == model._ep_experts_per_rank
        ep_out, ep_logits = model(inputs)

        diff = (ep_out - baseline_out).abs()
        diff_mean = diff.mean().item()
        diff_max = diff.max().item()
        if not torch.allclose(ep_out, baseline_out, rtol=1e-4, atol=1e-5):
            print(f"[rank{rank}] ep_out diff mean={diff_mean:.6e} max={diff_max:.6e}")
        assert torch.allclose(ep_out, baseline_out, rtol=1e-4, atol=1e-5)
        if not torch.allclose(ep_logits, baseline_logits, rtol=1e-4, atol=1e-5):
            logits_diff = (ep_logits - baseline_logits).abs()
            print(f"[rank{rank}] logits diff mean={logits_diff.mean().item():.6e} max={logits_diff.max().item():.6e}")
        assert torch.allclose(ep_logits, baseline_logits, rtol=1e-4, atol=1e-5)

        baseline_out.sum().backward()
        for expert in baseline.experts:
            if expert.weight.grad is not None:
                dist.all_reduce(expert.weight.grad, op=dist.ReduceOp.SUM)
        ep_out.sum().backward()

        assert torch.allclose(model.gate.weight.grad, baseline.gate.weight.grad, rtol=1e-4, atol=1e-5)
        for local_idx, expert in enumerate(model.experts):
            global_idx = model._ep_local_start + local_idx
            baseline_grad = baseline.experts[global_idx].weight.grad
            ep_grad = expert.weight.grad

            if baseline_grad is None:
                assert ep_grad is None
            else:
                assert ep_grad is not None, f"[rank{rank}] expert{global_idx}: baseline_grad is not None but ep_grad is None!"
                assert torch.allclose(ep_grad, baseline_grad, rtol=1e-4, atol=1e-5)
    finally:
        dist.destroy_process_group()


class TestExpertParallel(unittest.TestCase):
    def test_qwen3_moe_like_ep(self):
        if not dist.is_available():
            self.skipTest("torch.distributed is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for this test.")
        world_size = 2
        port = _find_free_port()
        mp.spawn(_run_worker, args=(world_size, port), nprocs=world_size, join=True)
