# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from typing import Any, Dict, Iterable, List, Optional, Tuple

from twinkle.model.transformers.moe.ep_utils import preprocess, token_pre_all2all, tokens_post_all2all
from twinkle.utils import DeviceMesh


@dataclass
class ExpertParallelConfig:
    enabled: bool = True
    router_dtype: str = 'fp32'
    keep_router_logits: bool = True
    ignore_shared_experts: bool = False
    ep_size: int | None = None  # consumed by TransformersModel, not used in expert_parallel logic


@dataclass
class ExpertShardingSpec:
    """Describes expert sharding info for a single MoE block. Extensible for other models."""
    block: nn.Module
    experts_module: nn.Module
    num_experts: int
    experts_per_rank: int
    local_start: int
    local_end: int
    ep_rank: int
    ep_world_size: int
    is_tensor_experts: bool


def apply_expert_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    config: dict[str, Any] | None = None,
    ep_fsdp_device_mesh: torch.distributed.DeviceMesh | None = None,
) -> list[ExpertShardingSpec]:
    """Apply expert parallelism to all MoE blocks in the model."""
    cfg = _merge_config(config)

    # EP info comes from the separate ep_fsdp_device_mesh, not from main mesh
    if not cfg.enabled or ep_fsdp_device_mesh is None:
        return []

    # Always query EP via the 1D submesh to avoid relying on Tensor named dims.
    ep_mesh = ep_fsdp_device_mesh['ep']
    ep_world_size = ep_mesh.size()
    if ep_world_size <= 1:
        return []

    if not dist.is_initialized():
        raise RuntimeError('torch.distributed is not initialized, cannot enable expert parallel.')

    # Get process group and local rank from EP submesh.
    ep_group = ep_mesh.get_group()
    ep_rank = ep_mesh.get_local_rank()

    specs = []
    for block in find_moe_blocks(model):
        spec = shard_experts(block, ep_world_size, ep_rank, cfg)
        patch_forward(block, ep_group, ep_world_size, cfg)
        specs.append(spec)

    return specs


def _merge_config(config: dict[str, Any] | None) -> ExpertParallelConfig:
    cfg = ExpertParallelConfig()
    if not config:
        return cfg
    for key, value in config.items():
        if not hasattr(cfg, key):
            raise ValueError(f'Unknown expert parallel config: {key}')
        setattr(cfg, key, value)
    return cfg


def find_moe_blocks(model: nn.Module) -> Iterable[nn.Module]:
    blocks = []
    for module in model.modules():
        experts = getattr(module, 'experts', None)
        if experts is None:
            continue
        if not _is_moe_experts(experts):
            continue
        if not _get_gate(module):
            continue
        blocks.append(module)
    return blocks


def shard_experts(
    block: nn.Module,
    ep_world_size: int,
    ep_rank: int,
    cfg: ExpertParallelConfig,
) -> ExpertShardingSpec:
    """Shard experts in a MoE block across EP ranks.

    Args:
        block: The MoE block containing experts.
        ep_world_size: The world size for expert parallelism.
        ep_rank: The current rank in the EP group.
        cfg: Expert parallel configuration.

    Returns an ExpertShardingSpec describing the sharding.
    """
    num_experts = _get_num_experts(block)

    if num_experts % ep_world_size != 0:
        raise ValueError(f'num_experts ({num_experts}) must be divisible by ep_world_size ({ep_world_size}).')

    experts_per_rank = num_experts // ep_world_size
    local_start = ep_rank * experts_per_rank
    local_end = local_start + experts_per_rank

    if isinstance(block.experts, nn.ModuleList):
        local_experts = nn.ModuleList(block.experts[local_start:local_end])
        block.experts = local_experts
        is_tensor_experts = False
    else:
        _shard_tensor_experts(block.experts, local_start, local_end)
        is_tensor_experts = True

    block._ep_num_experts = num_experts
    block._ep_experts_per_rank = experts_per_rank
    block._ep_local_start = local_start
    block._ep_local_end = local_end
    block._ep_rank = ep_rank
    block._ep_world_size = ep_world_size
    block._ep_tensor_experts = is_tensor_experts
    block._ep_ignore_shared_experts = cfg.ignore_shared_experts

    return ExpertShardingSpec(
        block=block,
        experts_module=block.experts,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        local_start=local_start,
        local_end=local_end,
        ep_rank=ep_rank,
        ep_world_size=ep_world_size,
        is_tensor_experts=is_tensor_experts,
    )


def patch_forward(
    block: nn.Module,
    ep_group: dist.ProcessGroup,
    ep_world_size: int,
    cfg: ExpertParallelConfig,
) -> None:
    """Replace the MoE block forward with EP-aware communication flow.

    Communication pattern:
        preprocess → token_pre_all2all → expert_compute → tokens_post_all2all

    For tensor experts (gate_up_proj/down_proj), the expert compute is delegated
    to block.experts(...) via nn.Module.__call__ so that FSDP2 pre/post-forward
    hooks fire correctly (automatic unshard before forward, backward hook
    registration, and reshard after forward). No manual unshard/reshard is needed.

    For ModuleList experts, each sub-expert is already called via __call__ inside
    _run_local_experts, so the same principle applies.

    Args:
        block: The MoE block to patch.
        ep_group: The process group for EP communication (from ep_fsdp_device_mesh["ep"]).
        ep_world_size: The world size for expert parallelism.
        cfg: Expert parallel configuration.
    """
    if getattr(block, '_ep_patched', False):
        return

    gate = _get_gate(block)
    if gate is None:
        raise ValueError('MoE block must define gate/router module.')

    top_k = _get_top_k(block)
    if top_k is None:
        raise ValueError('MoE block must define top_k/num_experts_per_tok.')

    orig_forward = block.forward
    num_experts = block._ep_num_experts
    experts_per_rank = block._ep_experts_per_rank
    is_tensor_experts = block._ep_tensor_experts

    # For tensor experts, install an ep_forward on the experts module so we can
    # call block.experts(permuted_tokens, counts, experts_per_rank) via __call__,
    # letting FSDP2 manage unshard/reshard automatically.
    if is_tensor_experts:
        _install_ep_forward(block.experts, experts_per_rank)

    def forward(hidden_states: torch.Tensor, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError('Expert parallel patch only supports forward(hidden_states).')

        orig_shape = hidden_states.shape
        if hidden_states.ndim == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_states_2d = hidden_states.view(-1, hidden_dim)
        elif hidden_states.ndim == 2:
            batch_size, seq_len = 1, hidden_states.shape[0]
            hidden_dim = hidden_states.shape[1]
            hidden_states_2d = hidden_states
        else:
            raise ValueError(f'Unsupported hidden_states ndim: {hidden_states.ndim}')

        router_logits, routing_weights, selected_experts = _run_router(
            gate=gate,
            hidden_states=hidden_states_2d,
            top_k=top_k,
            router_dtype=_get_router_dtype(cfg.router_dtype, hidden_states_2d.dtype),
            norm_topk_prob=getattr(block, 'norm_topk_prob', False),
        )
        # Keep routing weights in activation dtype before unpermute weighting.
        if routing_weights.dtype != hidden_states_2d.dtype:
            routing_weights = routing_weights.to(hidden_states_2d.dtype)

        # Build expert_mask: [num_experts, top_k, num_tokens]
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=num_experts).permute(2, 1, 0)  # [num_experts, top_k, num_tokens]

        # 1. preprocess: compute splits and token counts
        (
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            num_global_sum_tokens_per_local_expert,
        ) = preprocess(expert_mask, num_experts, ep_group)

        # 2. token_pre_all2all: permute → all_to_all → sort_chunks
        (
            global_permuted_hidden_states,
            routing_map,
            local_input_permutation_mapping,
            org_hidden_states_shape,
        ) = token_pre_all2all(
            hidden_states_2d,
            expert_mask,
            num_experts,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            ep_group,
        )

        # 3. expert_compute: call experts via nn.Module.__call__ so FSDP2 hooks fire.
        # For tensor experts: block.experts(permuted_tokens, counts, experts_per_rank)
        #   → FSDP2 pre-forward unshard → ep_forward → FSDP2 post-forward reshard
        # For ModuleList experts: _run_local_experts calls each expert[i](...) via __call__.
        if is_tensor_experts:
            expert_outputs = block.experts(
                global_permuted_hidden_states,
                num_global_sum_tokens_per_local_expert,
                experts_per_rank,
            )
        else:
            expert_outputs = _run_local_experts(
                block,
                global_permuted_hidden_states,
                num_global_sum_tokens_per_local_expert,
                experts_per_rank,
            )

        # 4. tokens_post_all2all: sort_chunks → all_to_all → unpermute (with routing weight)
        final_hidden = tokens_post_all2all(
            expert_outputs,
            routing_weights,
            selected_experts,
            num_experts,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            routing_map,
            local_input_permutation_mapping,
            org_hidden_states_shape,
            ep_group,
        )

        shared_out = _maybe_run_shared_expert(block, hidden_states_2d, cfg)
        if shared_out is not None:
            final_hidden = final_hidden + shared_out

        if len(orig_shape) == 3:
            final_hidden = final_hidden.view(batch_size, seq_len, hidden_dim)

        if cfg.keep_router_logits:
            return final_hidden, router_logits
        return final_hidden

    block._ep_original_forward = orig_forward
    block.forward = forward
    block._ep_patched = True


def _install_ep_forward(experts_mod: nn.Module, experts_per_rank: int) -> None:
    if getattr(experts_mod, '_ep_forward_installed', False):
        return

    def ep_forward(
        self,
        permuted_tokens: torch.Tensor,
        num_global_sum_tokens_per_local_expert: torch.Tensor,
        experts_per_rank: int,
    ) -> torch.Tensor:
        if permuted_tokens.numel() == 0:
            return torch.empty_like(permuted_tokens)

        input_dtype = permuted_tokens.dtype

        cumsum = torch.zeros(experts_per_rank + 1, dtype=torch.long)
        for i in range(experts_per_rank):
            cumsum[i + 1] = cumsum[i] + int(num_global_sum_tokens_per_local_expert[i].item())

        output_chunks = []
        for i in range(experts_per_rank):
            start = int(cumsum[i].item())
            end = int(cumsum[i + 1].item())
            expert_in = permuted_tokens[start:end]
            if expert_in.numel() == 0:
                output_chunks.append(expert_in)
                continue

            gate_up = self.gate_up_proj[i]
            down = self.down_proj[i]
            compute_dtype = gate_up.dtype
            if expert_in.dtype != compute_dtype:
                expert_in = expert_in.to(compute_dtype)
            gate, up = F.linear(expert_in, gate_up).chunk(2, dim=-1)
            out = self.act_fn(gate) * up
            out = F.linear(out, down)

            if out.dtype != input_dtype:
                out = out.to(input_dtype)
            output_chunks.append(out)

        return torch.cat(
            output_chunks, dim=0) if output_chunks else permuted_tokens.new_empty(0, permuted_tokens.size(-1))

    import types
    experts_mod.forward = types.MethodType(ep_forward, experts_mod)
    experts_mod._ep_forward_installed = True


def _get_gate(block: nn.Module):
    gate = getattr(block, 'gate', None)
    if gate is None:
        gate = getattr(block, 'router', None)
    return gate


def _get_num_experts(block: nn.Module) -> int:
    if hasattr(block, 'num_experts'):
        return int(block.num_experts)
    experts = getattr(block, 'experts', None)
    if experts is None:
        raise ValueError('MoE block has no experts.')
    if isinstance(experts, nn.ModuleList):
        return len(experts)
    if hasattr(experts, 'num_experts'):
        return int(experts.num_experts)
    if hasattr(experts, 'gate_up_proj'):
        return int(experts.gate_up_proj.shape[0])
    raise ValueError('Unable to infer num_experts for MoE block.')


def _get_top_k(block: nn.Module) -> int | None:
    gate = _get_gate(block)
    if gate is not None and hasattr(gate, 'top_k'):
        value = getattr(gate, 'top_k')
        if value is not None:
            return int(value)
    for name in ('num_experts_per_tok', 'top_k'):
        if hasattr(block, name):
            value = getattr(block, name)
            if value is not None:
                return int(value)
    return None


def _get_router_dtype(router_dtype: str, default_dtype: torch.dtype) -> torch.dtype:
    if router_dtype == 'fp32':
        return torch.float32
    if router_dtype == 'bf16':
        return torch.bfloat16
    if router_dtype == 'fp16':
        return torch.float16
    return default_dtype


def _maybe_run_shared_expert(block: nn.Module, hidden_states_2d: torch.Tensor, cfg: ExpertParallelConfig):
    if cfg.ignore_shared_experts:
        return None
    shared = getattr(block, 'shared_expert', None)
    if shared is None:
        return None
    return _run_module_with_casting(shared, hidden_states_2d)


def _is_moe_experts(experts: Any) -> bool:
    if isinstance(experts, nn.ModuleList):
        return True
    if hasattr(experts, 'gate_up_proj') and hasattr(experts, 'down_proj'):
        return True
    return False


def _shard_tensor_experts(experts: nn.Module, start: int, end: int) -> None:
    experts.gate_up_proj = nn.Parameter(experts.gate_up_proj.data[start:end].clone())
    experts.down_proj = nn.Parameter(experts.down_proj.data[start:end].clone())
    if hasattr(experts, 'num_experts'):
        experts.num_experts = end - start


def _run_local_experts(
    block: nn.Module,
    permuted_tokens: torch.Tensor,
    num_global_sum_tokens_per_local_expert: torch.Tensor,
    experts_per_rank: int,
) -> torch.Tensor:
    """Run ModuleList experts on permuted tokens via nn.Module.__call__.
    Tokens are already grouped by expert (contiguous chunks), sizes given by
    num_global_sum_tokens_per_local_expert. No routing weight is applied here;
    that happens in unpermute.
    """
    if permuted_tokens.numel() == 0:
        return torch.empty_like(permuted_tokens)

    input_dtype = permuted_tokens.dtype
    experts = block.experts

    cumsum = torch.zeros(experts_per_rank + 1, dtype=torch.long)
    for i in range(experts_per_rank):
        cumsum[i + 1] = cumsum[i] + int(num_global_sum_tokens_per_local_expert[i].item())

    output_chunks = []
    for i in range(experts_per_rank):
        start = int(cumsum[i].item())
        end = int(cumsum[i + 1].item())
        expert_in = permuted_tokens[start:end]
        if expert_in.numel() == 0:
            output_chunks.append(expert_in)
            continue

        expert = experts[i]
        compute_dtype = _module_compute_dtype(expert, input_dtype)
        if expert_in.dtype != compute_dtype:
            expert_in = expert_in.to(compute_dtype)
        out = expert(expert_in)

        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        output_chunks.append(out)

    return torch.cat(output_chunks, dim=0) if output_chunks else permuted_tokens.new_empty(0, permuted_tokens.size(-1))


def _module_compute_dtype(module: nn.Module, default: torch.dtype) -> torch.dtype:
    for param in module.parameters():
        if param.dtype.is_floating_point:
            return param.dtype
    return default


def _run_module_with_casting(module: nn.Module, module_in: torch.Tensor) -> torch.Tensor:
    input_dtype = module_in.dtype
    compute_dtype = _module_compute_dtype(module, input_dtype)
    if compute_dtype != input_dtype:
        module_in = module_in.to(compute_dtype)
    out = module(module_in)
    if out.dtype != input_dtype:
        out = out.to(input_dtype)
    return out


def _run_router(
    *,
    gate: nn.Module,
    hidden_states: torch.Tensor,
    top_k: int,
    router_dtype: torch.dtype,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_out = gate(hidden_states)
    if isinstance(gate_out, tuple) and len(gate_out) >= 3:
        router_logits, routing_weights, selected_experts = gate_out[:3]
        return router_logits, routing_weights, selected_experts

    router_logits = gate_out
    routing_weights = torch.softmax(router_logits, dim=-1, dtype=router_dtype)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    return router_logits, routing_weights, selected_experts
