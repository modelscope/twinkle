# Copyright (c) ModelScope Contributors. All rights reserved.
"""MoE GMM + packed-experts + sparse-block impls for Ascend NPU."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GmmFunction(torch.autograd.Function):
    """Custom autograd function for NPU grouped matrix multiplication."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_list: torch.Tensor, weight_ekn: torch.Tensor):
        import torch_npu
        group_list = group_list.to(torch.int64)
        ctx.save_for_backward(x, group_list, weight_ekn)
        outputs = torch_npu.npu_grouped_matmul(
            [x], [weight_ekn], group_list=group_list,
            group_type=0, split_item=2, group_list_type=1,
        )
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        import torch_npu
        x, group_list, weight_ekn = ctx.saved_tensors
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output], [weight_ekn.transpose(-2, -1).contiguous()],
            bias=None, group_list=group_list,
            group_type=0, split_item=2, group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [x.transpose(0, 1)], [grad_output],
            bias=None, group_list=group_list,
            group_type=2, split_item=3, group_list_type=1,
        )[0]
        return grad_input, None, grad_weight.contiguous()


def npu_grouped_mm(input: torch.Tensor, weight_ekn: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for ``transformers.integrations.moe._grouped_mm``."""
    counts = torch.empty_like(offs)
    counts[0] = offs[0]
    if offs.numel() > 1:
        counts[1:] = offs[1:] - offs[:-1]
    counts = counts.to(torch.int64)
    return GmmFunction.apply(input, counts, weight_ekn)


def _normalize_packed_expert_weights(module, input_dtype, hidden_dim):
    gate_up_proj = module.gate_up_proj.to(input_dtype)
    down_proj = module.down_proj.to(input_dtype)
    if gate_up_proj.shape[1] == hidden_dim:
        gate_up_weight = gate_up_proj
    elif gate_up_proj.shape[2] == hidden_dim:
        gate_up_weight = gate_up_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported gate_up_proj shape: {tuple(gate_up_proj.shape)}.')
    if down_proj.shape[2] == hidden_dim:
        down_weight = down_proj
    elif down_proj.shape[1] == hidden_dim:
        down_weight = down_proj.transpose(1, 2)
    else:
        raise RuntimeError(f'Unsupported down_proj shape: {tuple(down_proj.shape)}.')
    return gate_up_weight, down_weight


def _get_cached_expert_weights(self, target_dtype, hidden_dim):
    requires_grad = (
        getattr(self.gate_up_proj, 'requires_grad', False)
        or getattr(self.down_proj, 'requires_grad', False)
    )
    cache_attr = '_npu_expert_cache'
    if not requires_grad and hasattr(self, cache_attr):
        cached_dtype, cached_gv, cached_dv, cached = getattr(self, cache_attr)
        if (cached_dtype == target_dtype
                and cached_gv == self.gate_up_proj._version
                and cached_dv == self.down_proj._version):
            return cached
    weights = _normalize_packed_expert_weights(self, target_dtype, hidden_dim)
    if not requires_grad:
        setattr(self, cache_attr,
                (target_dtype, self.gate_up_proj._version, self.down_proj._version, weights))
    return weights


def npu_packed_moe_experts_forward(self, hidden_states, a, b):
    """Packed MoE Experts.forward using NPU grouped matmul.

    Accepts both call orderings: ``(hidden_states, routing_weights, router_indices)``
    and ``(hidden_states, router_indices, routing_weights)`` — distinguishes by dtype.
    """
    import torch_npu
    if a.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        router_indices, routing_weights = a, b
    else:
        routing_weights, router_indices = a, b

    output_shape = hidden_states.shape
    hidden_dim = output_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)

    if routing_weights.shape != router_indices.shape:
        routing_weights = torch.gather(routing_weights, dim=-1, index=router_indices.to(torch.long))
    routing_weights = routing_weights.to(hidden_states.dtype)
    router_indices = router_indices.to(torch.int32)

    permuted, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, router_indices)
    tokens_per_expert = torch.bincount(router_indices.view(-1), minlength=self.num_experts).to(torch.int64)
    gate_up_weight, down_weight = _get_cached_expert_weights(self, hidden_states.dtype, hidden_dim)

    intermediate = GmmFunction.apply(permuted, tokens_per_expert, gate_up_weight)
    activated = torch_npu.npu_swiglu(intermediate, dim=-1)
    output = GmmFunction.apply(activated, tokens_per_expert, down_weight)
    next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
    return next_states.view(*output_shape)


def _topk_from_router_logits(module, hidden_states, router_logits):
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, router_indices = torch.topk(routing_weights, module.top_k, dim=-1)
    if getattr(module, 'norm_topk_prob', True):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    return routing_weights, router_indices


def _add_shared_expert(self, hidden_states, expert_output):
    if not (hasattr(self, 'shared_expert') and hasattr(self, 'shared_expert_gate')):
        return expert_output
    shared = self.shared_expert(hidden_states)
    shared = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared
    return expert_output + shared


def npu_qwen3_5_moe_sparse_block_forward(self, hidden_states):
    """SparseMoeBlock.forward replacement (Transformers 4.x and 5.x compatible)."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    gate_output = self.gate(hidden_states.view(-1, hidden_dim))

    if isinstance(gate_output, tuple):
        _, routing_weights, selected_experts = gate_output
        flat = hidden_states.view(-1, hidden_dim)
        expert_output = self.experts(flat, selected_experts, routing_weights)
    else:
        flat = hidden_states.view(-1, hidden_dim)
        routing_weights, selected_experts = _topk_from_router_logits(self, flat, gate_output)
        expert_output = self.experts(flat, selected_experts, routing_weights)

    expert_output = _add_shared_expert(self, flat, expert_output)
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)