#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
CPU script to verify numerical alignment between:
1) transformers original Qwen3MoeSparseMoeBlock forward
2) twinkle local path: permute -> expert compute -> unpermute

This script does NOT use distributed init and does NOT include all_to_all communication.

Usage:
    python tests/moe/verify_qwen3_moe_permute_cpu.py
    python tests/moe/verify_qwen3_moe_permute_cpu.py --transformers-root /mnt/d/workspace/transformers
"""

from __future__ import annotations
from twinkle.model.transformers.moe.expert_parallel import _run_router
from twinkle.model.transformers.moe.ep_utils import generate_weights_idx, permute, unpermute

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# Allow running directly from repository root without installing twinkle.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _import_qwen3_block(transformers_root: str):
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        return Qwen3MoeSparseMoeBlock
    except Exception:
        src_dir = Path(transformers_root).expanduser().resolve() / 'src'
        if not src_dir.exists():
            raise RuntimeError(
                f'Cannot import transformers qwen3_moe, and fallback path does not exist: {src_dir}'
            )
        sys.path.insert(0, str(src_dir))
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        return Qwen3MoeSparseMoeBlock


def _build_config(
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    top_k: int,
    hidden_act: str,
    norm_topk_prob: bool,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_act=hidden_act,
        norm_topk_prob=norm_topk_prob,
        _experts_implementation='eager',
    )


def _run_local_permute_expert_unpermute(block: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Reproduce Twinkle EP compute path locally without communication."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    hidden_states_2d = hidden_states.view(-1, hidden_dim)

    _, routing_weights, selected_experts = _run_router(
        gate=block.gate,
        hidden_states=hidden_states_2d,
        top_k=block.gate.top_k,
        router_dtype=torch.float32,
        norm_topk_prob=bool(getattr(block.gate, 'norm_topk_prob', False)),
    )

    num_experts = int(block.experts.num_experts)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    routing_map = expert_mask.sum(dim=1)

    permuted_tokens, permutation_mapping = permute(hidden_states_2d, routing_map)

    num_tokens_per_expert = routing_map.sum(dim=1).to(dtype=torch.long)
    cumsum = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=num_tokens_per_expert.device),
            num_tokens_per_expert.cumsum(dim=0),
        ]
    )

    outputs = []
    experts = block.experts
    input_dtype = permuted_tokens.dtype

    for expert_idx in range(num_experts):
        start = int(cumsum[expert_idx].item())
        end = int(cumsum[expert_idx + 1].item())
        x = permuted_tokens[start:end]
        if x.numel() == 0:
            outputs.append(x)
            continue

        gate_up = experts.gate_up_proj[expert_idx]
        down = experts.down_proj[expert_idx]
        compute_dtype = gate_up.dtype

        if x.dtype != compute_dtype:
            x = x.to(compute_dtype)

        gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
        out = experts.act_fn(gate) * up
        out = F.linear(out, down)

        if out.dtype != input_dtype:
            out = out.to(input_dtype)

        outputs.append(out)

    expert_outputs = (
        torch.cat(outputs, dim=0)
        if outputs
        else permuted_tokens.new_empty((0, permuted_tokens.size(-1)))
    )

    weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)
    final_hidden_states_2d = unpermute(
        expert_outputs,
        weights_idx,
        hidden_states_2d.shape,
        permutation_mapping,
        routing_map,
    )

    return final_hidden_states_2d.view(batch_size, seq_len, hidden_dim)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@torch.no_grad()
def _init_block_weights(block: torch.nn.Module, seed: int, std: float = 0.02) -> None:
    # Standalone Qwen3MoeSparseMoeBlock does not go through PreTrainedModel.post_init(),
    # so initialize weights explicitly to avoid trivial all-zero / allocator-dependent cases.
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    experts = block.experts
    gate = block.gate
    experts.gate_up_proj.copy_(torch.randn(experts.gate_up_proj.shape, generator=generator) * std)
    experts.down_proj.copy_(torch.randn(experts.down_proj.shape, generator=generator) * std)
    gate.weight.copy_(torch.randn(gate.weight.shape, generator=generator) * std)


def main():
    parser = argparse.ArgumentParser(description='CPU precision check for Qwen3-MoE sparse block.')
    parser.add_argument('--transformers-root', type=str, default='/mnt/d/workspace/transformers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--moe-intermediate-size', type=int, default=32)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--hidden-act', type=str, default='silu')
    parser.add_argument('--norm-topk-prob', action='store_true')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-6)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    Qwen3MoeSparseMoeBlock = _import_qwen3_block(args.transformers_root)
    cfg = _build_config(
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_act=args.hidden_act,
        norm_topk_prob=args.norm_topk_prob,
    )

    ref_block = Qwen3MoeSparseMoeBlock(cfg).cpu().float().train()
    _init_block_weights(ref_block, seed=args.seed)
    test_block = copy.deepcopy(ref_block).cpu().float().train()

    hidden_ref = torch.randn(args.batch_size, args.seq_len, args.hidden_size, dtype=torch.float32, requires_grad=True)
    hidden_test = hidden_ref.detach().clone().requires_grad_(True)

    ref_out = ref_block(hidden_ref)
    test_out = _run_local_permute_expert_unpermute(test_block, hidden_test)

    # Use identical loss form for backward alignment.
    proj = torch.randn(args.hidden_size, dtype=torch.float32)
    ref_loss = (ref_out * proj).sum()
    test_loss = (test_out * proj).sum()

    ref_loss.backward()
    test_loss.backward()

    out_max_diff = _max_abs_diff(ref_out.detach(), test_out.detach())
    in_grad_max_diff = _max_abs_diff(hidden_ref.grad.detach(), hidden_test.grad.detach())

    print('\n=== Qwen3-MoE Sparse Block CPU Alignment ===')
    print(f'seed={args.seed} shape=({args.batch_size}, {args.seq_len}, {args.hidden_size})')
    print(f'num_experts={args.num_experts} top_k={args.top_k} hidden_act={args.hidden_act}')
    print(f'forward max_abs_diff: {out_max_diff:.8e}')
    print(f'input grad max_abs_diff: {in_grad_max_diff:.8e}')

    param_ok = True
    for (name_ref, p_ref), (name_test, p_test) in zip(ref_block.named_parameters(), test_block.named_parameters()):
        if name_ref != name_test:
            raise RuntimeError(f'Parameter name mismatch: {name_ref} vs {name_test}')
        if p_ref.grad is None or p_test.grad is None:
            raise RuntimeError(f'Missing grad for parameter: {name_ref}')
        diff = _max_abs_diff(p_ref.grad.detach(), p_test.grad.detach())
        print(f'grad[{name_ref}] max_abs_diff: {diff:.8e}')
        if not torch.allclose(p_ref.grad, p_test.grad, rtol=args.rtol, atol=args.atol):
            param_ok = False

    out_ok = torch.allclose(ref_out, test_out, rtol=args.rtol, atol=args.atol)
    in_grad_ok = torch.allclose(hidden_ref.grad, hidden_test.grad, rtol=args.rtol, atol=args.atol)

    print('\n=== Result ===')
    print(f'forward aligned: {out_ok}')
    print(f'input grad aligned: {in_grad_ok}')
    print(f'param grad aligned: {param_ok}')

    if not (out_ok and in_grad_ok and param_ok):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
