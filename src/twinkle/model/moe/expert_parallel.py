# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import nn as dist_nn
from torch import nn

from ...utils import DeviceMesh


@dataclass
class ExpertParallelConfig:
    enabled: bool = True
    router_dtype: str = "fp32"
    all_to_all: str = "torch"
    keep_router_logits: bool = True
    pad_to_max: bool = False
    ignore_shared_experts: bool = False


def apply_expert_parallel(model: nn.Module, device_mesh: DeviceMesh, config: Optional[Dict[str, Any]] = None):
    cfg = _merge_config(config)
    if not cfg.enabled or device_mesh is None or not device_mesh.has_dim("ep"):
        return model

    ep_world_size = device_mesh.ep_world_size
    if ep_world_size <= 1:
        return model

    if cfg.pad_to_max:
        raise NotImplementedError("pad_to_max is not implemented.")
    if cfg.all_to_all != "torch":
        raise NotImplementedError(f"all_to_all={cfg.all_to_all} is not supported.")

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized, cannot enable expert parallel.")

    ep_group = device_mesh.get_dim_group("ep")
    if ep_group is None:
        raise RuntimeError("EP process group is not available in device_mesh.")

    for block in find_moe_blocks(model):
        shard_experts(block, device_mesh, cfg)
        patch_forward(block, device_mesh, cfg)

    return model


def _merge_config(config: Optional[Dict[str, Any]]) -> ExpertParallelConfig:
    cfg = ExpertParallelConfig()
    if not config:
        return cfg
    for key, value in config.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown expert parallel config: {key}")
        setattr(cfg, key, value)
    return cfg


def find_moe_blocks(model: nn.Module) -> Iterable[nn.Module]:
    blocks = []
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        if not _is_moe_experts(experts):
            continue
        if not _get_gate(module):
            continue
        blocks.append(module)
    return blocks


def shard_experts(block: nn.Module, device_mesh: DeviceMesh, cfg: ExpertParallelConfig) -> None:
    num_experts = _get_num_experts(block)
    ep_world_size = device_mesh.ep_world_size
    ep_rank = device_mesh.ep_rank

    if num_experts % ep_world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_world_size ({ep_world_size})."
        )

    experts_per_rank = num_experts // ep_world_size
    local_start = ep_rank * experts_per_rank
    local_end = local_start + experts_per_rank

    if isinstance(block.experts, nn.ModuleList):
        local_experts = nn.ModuleList(block.experts[local_start:local_end])
        block.experts = local_experts
        block._ep_tensor_experts = False
    else:
        _shard_tensor_experts(block.experts, local_start, local_end)
        block._ep_tensor_experts = True

    block._ep_num_experts = num_experts
    block._ep_experts_per_rank = experts_per_rank
    block._ep_local_start = local_start
    block._ep_local_end = local_end
    block._ep_rank = ep_rank
    block._ep_world_size = ep_world_size
    block._ep_ignore_shared_experts = cfg.ignore_shared_experts


def patch_forward(block: nn.Module, device_mesh: DeviceMesh, cfg: ExpertParallelConfig) -> None:
    if getattr(block, "_ep_patched", False):
        return

    gate = _get_gate(block)
    if gate is None:
        raise ValueError("MoE block must define gate/router module.")

    top_k = _get_top_k(block)
    if top_k is None:
        raise ValueError("MoE block must define top_k/num_experts_per_tok.")

    orig_forward = block.forward
    ep_group = device_mesh.get_dim_group("ep")

    def forward(hidden_states: torch.Tensor, *args, **kwargs):
        ep_rank = block._ep_rank
        if args or kwargs:
            raise RuntimeError("Expert parallel patch only supports forward(hidden_states).")

        input_dtype = hidden_states.dtype
        if hidden_states.ndim == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, hidden_dim)
        elif hidden_states.ndim == 2:
            batch_size, seq_len = 1, hidden_states.shape[0]
            hidden_dim = hidden_states.shape[1]
            hidden_states_2d = hidden_states
        else:
            raise ValueError(f"Unsupported hidden_states ndim: {hidden_states.ndim}")

        router_logits, routing_weights, selected_experts, cast_weights = _run_router(
            gate=gate,
            hidden_states=hidden_states_2d,
            top_k=top_k,
            router_dtype=_get_router_dtype(cfg.router_dtype, hidden_states_2d.dtype),
            norm_topk_prob=getattr(block, "norm_topk_prob", False),
        )
        if cast_weights:
            routing_weights = routing_weights.to(hidden_states_2d.dtype)

        num_tokens = hidden_states_2d.shape[0]
        flat_token_idx = torch.arange(num_tokens, device=hidden_states_2d.device).repeat_interleave(top_k)
        flat_expert_id = selected_experts.reshape(-1)
        flat_weight = routing_weights.reshape(-1)

        experts_per_rank = block._ep_experts_per_rank
        dest_rank = flat_expert_id // experts_per_rank
        local_expert_id = flat_expert_id - dest_rank * experts_per_rank

        order = torch.argsort(dest_rank)
        ordered_token_idx = flat_token_idx[order]
        ordered_weight = flat_weight[order]
        ordered_global_expert_id = flat_expert_id[order]
        ordered_expert_id = local_expert_id[order]

        send_counts = torch.bincount(dest_rank, minlength=block._ep_world_size)
        send_counts_list = send_counts.cpu().tolist()

        recv_counts = _exchange_counts(send_counts, ep_group)
        recv_counts_list = recv_counts.cpu().tolist()

        send_tokens = hidden_states_2d.index_select(0, ordered_token_idx)
        recv_tokens = torch.empty(
            (int(recv_counts.sum().item()), hidden_dim),
            device=hidden_states_2d.device,
            dtype=hidden_states_2d.dtype,
        )
        send_expert_ids = ordered_expert_id.to(torch.int64)
        recv_expert_ids = torch.empty(
            (int(recv_counts.sum().item()),),
            device=hidden_states_2d.device,
            dtype=torch.int64,
        )

        recv_tokens = dist_nn.functional.all_to_all_single(
            recv_tokens,
            send_tokens,
            input_split_sizes=send_counts_list,
            output_split_sizes=recv_counts_list,
            group=ep_group,
        )
        dist.all_to_all_single(
            recv_expert_ids,
            send_expert_ids.to(torch.int64),
            input_split_sizes=send_counts_list,
            output_split_sizes=recv_counts_list,
            group=ep_group,
        )
        recv_out = torch.empty_like(recv_tokens)
        for expert_id in torch.unique(recv_expert_ids).tolist():
            idx = (recv_expert_ids == expert_id).nonzero(as_tuple=False).view(-1)
            expert_in = recv_tokens.index_select(0, idx)
            expert_out = _run_expert(block, expert_id, expert_in)
            recv_out.index_copy_(0, idx, expert_out)

        send_out = torch.empty_like(send_tokens)
        send_out = dist_nn.functional.all_to_all_single(
            send_out,
            recv_out,
            input_split_sizes=recv_counts_list,
            output_split_sizes=send_counts_list,
            group=ep_group,
        )

        final_hidden = torch.zeros(
            (num_tokens, hidden_dim), device=hidden_states_2d.device, dtype=input_dtype
        )
        expert_hit = torch.unique(ordered_global_expert_id)
        if expert_hit.numel() > 0:
            expert_hit, _ = torch.sort(expert_hit)
        for expert_id in expert_hit:
            idx = (ordered_global_expert_id == expert_id).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            token_idx = ordered_token_idx.index_select(0, idx)
            weight = ordered_weight.index_select(0, idx)
            contrib = send_out.index_select(0, idx)
            scaled = contrib * weight.unsqueeze(-1)
            final_hidden.index_add_(0, token_idx, scaled.to(input_dtype))

        shared_out = _maybe_run_shared_expert(block, hidden_states_2d, cfg)
        if shared_out is not None:
            final_hidden = final_hidden + shared_out

        if hidden_states.ndim == 3:
            final_hidden = final_hidden.view(batch_size, seq_len, hidden_dim)

        if cfg.keep_router_logits and not getattr(block, "_ep_tensor_experts", False):
            return final_hidden, router_logits
        return final_hidden

    block._ep_original_forward = orig_forward
    block.forward = forward
    block._ep_patched = True


def _exchange_counts(send_counts: torch.Tensor, group) -> torch.Tensor:
    ep_world_size = int(send_counts.numel())
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(
        recv_counts,
        send_counts.to(torch.int64),
        input_split_sizes=[1] * ep_world_size,
        output_split_sizes=[1] * ep_world_size,
        group=group,
    )
    return recv_counts


def _get_gate(block: nn.Module):
    gate = getattr(block, "gate", None)
    if gate is None:
        gate = getattr(block, "router", None)
    return gate


def _get_num_experts(block: nn.Module) -> int:
    if hasattr(block, "num_experts"):
        return int(block.num_experts)
    experts = getattr(block, "experts", None)
    if experts is None:
        raise ValueError("MoE block has no experts.")
    if isinstance(experts, nn.ModuleList):
        return len(experts)
    if hasattr(experts, "num_experts"):
        return int(experts.num_experts)
    if hasattr(experts, "gate_up_proj"):
        return int(experts.gate_up_proj.shape[0])
    raise ValueError("Unable to infer num_experts for MoE block.")


def _get_top_k(block: nn.Module) -> Optional[int]:
    gate = _get_gate(block)
    if gate is not None and hasattr(gate, "top_k"):
        value = getattr(gate, "top_k")
        if value is not None:
            return int(value)
    for name in ("num_experts_per_tok", "top_k"):
        if hasattr(block, name):
            value = getattr(block, name)
            if value is not None:
                return int(value)
    return None


def _get_router_dtype(router_dtype: str, default_dtype: torch.dtype) -> torch.dtype:
    if router_dtype == "fp32":
        return torch.float32
    if router_dtype == "bf16":
        return torch.bfloat16
    if router_dtype == "fp16":
        return torch.float16
    return default_dtype


def _maybe_run_shared_expert(block: nn.Module, hidden_states_2d: torch.Tensor, cfg: ExpertParallelConfig):
    if cfg.ignore_shared_experts:
        return None
    shared = getattr(block, "shared_expert", None)
    if shared is None:
        return None
    return shared(hidden_states_2d)


def _is_moe_experts(experts: Any) -> bool:
    if isinstance(experts, nn.ModuleList):
        return True
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return True
    return False


def _shard_tensor_experts(experts: nn.Module, start: int, end: int) -> None:
    experts.gate_up_proj = nn.Parameter(experts.gate_up_proj.data[start:end].clone())
    experts.down_proj = nn.Parameter(experts.down_proj.data[start:end].clone())
    if hasattr(experts, "num_experts"):
        experts.num_experts = end - start


def _run_expert(block: nn.Module, expert_id: int, expert_in: torch.Tensor) -> torch.Tensor:
    if not getattr(block, "_ep_tensor_experts", False):
        return block.experts[expert_id](expert_in)
    experts = block.experts
    gate_up = experts.gate_up_proj[expert_id]
    down = experts.down_proj[expert_id]
    gate, up = F.linear(expert_in, gate_up).chunk(2, dim=-1)
    out = experts.act_fn(gate) * up
    return F.linear(out, down)


def _run_router(
    *,
    gate: nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    router_dtype: torch.dtype,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    gate_out = gate(hidden_states)
    if isinstance(gate_out, tuple) and len(gate_out) >= 3:
        router_logits, routing_weights, selected_experts = gate_out[:3]
        return router_logits, routing_weights, selected_experts, False

    router_logits = gate_out
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=router_dtype)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return router_logits, routing_weights, selected_experts, True
