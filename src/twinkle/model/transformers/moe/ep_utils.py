# Copyright (c) ModelScope Contributors. All rights reserved.
#
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0
# Adapted from https://github.com/ByteDance-Seed/VeOmni/tree/main/veomni/distributed/moe

import torch
import torch.distributed as dist
from typing import Optional


# ========================== comm ==========================
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


class _AllToAll_Async(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        async_handle = dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        return output, async_handle

    @staticmethod
    def backward(ctx, grad_output, grad_async_handle):
        return (
            None,
            _AllToAll_Async.apply(ctx.group, grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)


def all_to_all_async(group, input, output_split_size, input_split_size):
    return _AllToAll_Async.apply(group, input, output_split_size, input_split_size)


# ========================== moe_utils ==========================
def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    """
    Permutes the tokens according to the routing map.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    """
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.bool()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    tokens: torch.Tensor,
    routing_weights: torch.Tensor,
    hidden_states_shape: torch.Size,
    permutation_mapping: torch.Tensor,
    routing_map: torch.Tensor,
):
    """
    Unpermutes the tokens and apply the weight.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden_dim].
        routing_weights (torch.Tensor): The routing weights, [num_tokens, num_experts].
        hidden_states_shape (torch.Size): The shape of the hidden states, [num_tokens, hidden_dim].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_experts, tokens].

    Returns:
        torch.Tensor: The unpermuted token tensor, [num_tokens, hidden_dim].
    """
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())

    tokens = tokens * tokens_weight.unsqueeze(-1)
    hidden_dim = hidden_states_shape[-1]

    unpermuted_tokens = torch.zeros(hidden_states_shape, device=tokens.device, dtype=tokens.dtype)

    # Scatter add the permuted_input back to the original positions
    unpermuted_tokens.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens)
    return unpermuted_tokens


def generate_weights_idx(routing_weights: torch.Tensor, selected_experts: torch.Tensor, num_experts) -> torch.Tensor:
    """
    Generate the weight index for the unpermute operation.

    Args:
        routing_weights (torch.Tensor): The routing weights. shape [num_tokens, topk].
        selected_experts (torch.Tensor): The selected experts. shape [num_tokens, topk].
        num_experts (int): The number of experts. shape [num_tokens, num_experts].

    Returns:
        torch.Tensor: The weight index.
    """
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), dtype=routing_weights.dtype, device=routing_weights.device)

    weights_idx.scatter_add_(1, selected_experts, routing_weights)

    return weights_idx


def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    """Split and sort the input tensor based on the split_sizes and sorted indices."""
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output


# ========================== moe_layer ==========================
# EPGroupGemm is not included here; Twinkle uses an F.linear loop instead.


def preprocess(
    expert_mask: torch.Tensor,
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))

    # [ep_size] represent the number of sum tokens in each rank
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()

    # gather all the number of tokens per expert from all ep ranks
    # [ep_size, num_experts]
    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_local_tokens_per_expert.size(0),
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device,
    )
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    # [ep_size, num_local_experts]
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    # [ep_size]
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    # [num_local_expert]
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device('cpu'), non_blocking=True)

    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        torch.device('cpu'), non_blocking=True)

    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def token_pre_all2all(
    hidden_states: torch.Tensor,
    expert_mask: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    org_hidden_states_shape = hidden_states.shape
    routing_map = expert_mask.sum(dim=1)

    local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)

    global_permuted_hidden_states = all_to_all(ep_group, local_permuted_hidden_states, output_splits, input_splits)

    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
    global_permuted_hidden_states = sort_chunks_by_idxs(
        global_permuted_hidden_states,
        num_global_tokens_per_local_expert.ravel(),
        permute_order,
    )

    return global_permuted_hidden_states, routing_map, local_input_permutation_mapping, org_hidden_states_shape


def tokens_post_all2all(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_global_tokens_per_local_expert: torch.Tensor,
    routing_map: torch.Tensor,
    local_input_permutation_mapping: torch.Tensor,
    org_hidden_states_shape: torch.Size,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    # group tokens together by expert
    num_local_experts = num_experts // ep_group.size()
    unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
    expert_outputs = sort_chunks_by_idxs(
        expert_outputs,
        num_global_tokens_per_local_expert.T.ravel(),
        unpermute_order,
    )

    unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)

    # [tokens, experts]
    weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)

    unpermute_outputs = unpermute(
        unpermute_outputs,
        weights_idx,
        org_hidden_states_shape,
        local_input_permutation_mapping,
        routing_map,
    )

    return unpermute_outputs
