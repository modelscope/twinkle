# Copyright (c) ModelScope Contributors. All rights reserved.
"""
FSDP routing replay utilities for HF Transformers MoE models.

Provides RouterReplayAction, per-MoE-block replay state, and functions for
recording / replaying expert routing decisions during GRPO training.
"""

from __future__ import annotations

import inspect
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from twinkle import Platform
from twinkle.utils import get_logger

logger = get_logger()


class RouterReplayAction(Enum):
    """A Enum to define the actions for router replay."""

    RECORD = 'record'  # Record the topk indices for replay
    REPLAY_FORWARD = 'replay_forward'  # Replay the recorded topk indices for forward pass
    REPLAY_BACKWARD = 'replay_backward'  # Replay topk indices for re-compute during backward pass


# ---------------------------------------------------------------------------
# Global registry: {block_name: _RouterReplayState}
# ---------------------------------------------------------------------------


@dataclass
class _RouterReplayState:
    """Per-MoE-block replay state, stored in the global ``_registry``."""

    action: RouterReplayAction = None
    recorded_indices: torch.Tensor | None = None  # [num_tokens, topk]
    target_indices: torch.Tensor | None = None  # [num_tokens, topk]


_registry: dict[str, _RouterReplayState] = {}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_global_router_replay_action(action: RouterReplayAction) -> None:
    """Set *action* on every registered MoE block."""
    for state in _registry.values():
        state.action = action


def clear_global_router_replay_action() -> None:
    """Reset action to None on every registered MoE block."""
    for state in _registry.values():
        state.action = None


def clear_global_indices() -> None:
    """Clear recorded / target indices on every registered MoE block."""
    for state in _registry.values():
        state.recorded_indices = None
        state.target_indices = None


def get_replay_state(block_name: str) -> _RouterReplayState | None:
    """Return the replay state for *block_name*, or *None*."""
    return _registry.get(block_name)


def set_router_replay_data(
    routed_experts: torch.Tensor,
    model: nn.Module,
) -> None:
    """Slice *routed_experts* into per-block
    ``target_indices`` and inject them into the registered MoE blocks of *model*.

    Each block receives a ``[num_tokens, topk]`` slice covering the tokens
    processed by that layer.
    """
    if routed_experts is None:
        return

    blocks = _find_moe_blocks_with_names(model)
    if not blocks:
        return

    if routed_experts.dim() != 4:
        raise ValueError(f'Expected routed_experts with shape [bs, seq_len, layers, topk], '
                         f'got {tuple(routed_experts.shape)}')

    # SP: slice full-sequence routed_experts to local SP rank tokens.
    from ..strategy.sequence_parallel import sequence_parallel as sp
    sp_world_size = getattr(sp, 'sp_world_size', None) or 1
    if sp_world_size > 1:
        _, _, _, _, _, _, extra_values = sp.pad_and_split_inputs(
            None,
            None,
            None,
            None,
            None,
            None,
            real_position_ids=sp.real_position_ids,
            extra_split_values=[(routed_experts, 0, 1)])
        routed_experts = extra_values[0]

    # [bs, seq_len, num_moe_layers, topk] -> [total_seq, num_moe_layers, topk]
    routed_experts = routed_experts.flatten(0, 1).to(Platform.get_local_device())

    num_layers_in_data = routed_experts.shape[1]

    for layer_idx, (name, _) in enumerate(blocks):
        state = _registry.get(name)
        if state is None:
            continue
        if layer_idx >= num_layers_in_data:
            break

        # Each layer gets [num_tokens, topk]
        target = routed_experts[:, layer_idx, :].to(torch.int64)
        if target.numel() > 0:
            state.target_indices = target


def get_router_replay_data(model: nn.Module, batch_size=1) -> torch.Tensor | None:
    """Collect ``recorded_indices`` from all registered MoE blocks in *model*.

    . note::
        *ep_group* is accepted for signature parity with the Megatron
        version but is unused in FSDP — routing runs before the EP
        all-to-all so every EP rank records the same indices.

    Returns *None* when no MoE blocks have recorded routing data.
    """
    blocks = _find_moe_blocks_with_names(model)
    if not blocks:
        return None

    layers = []
    for name, _ in blocks:
        state = _registry.get(name)
        if state is not None and state.recorded_indices is not None:
            layers.append(state.recorded_indices)  # each: [num_tokens, topk]

    if not layers:
        return None

    # Stack: [num_tokens, num_layers, topk]
    routed_experts = torch.stack(layers, dim=1)
    _, num_layers, topk = routed_experts.shape
    # [num_tokens, num_layers, topk] -> [bs, local_seq_len, num_layers, topk]
    routed_experts = routed_experts.reshape(batch_size, -1, num_layers, topk)

    # SP: all-gather local routing data across SP ranks along seq_len dim.
    from ..strategy.sequence_parallel import sequence_parallel as sp
    sp_world_size = getattr(sp, 'sp_world_size', None) or 1
    if sp_world_size > 1:
        routed_experts = sp.gather(routed_experts, dim=1, position_ids=sp.real_position_ids)

    # [bs, seq_len, num_layers, topk]
    return routed_experts


def apply_router_replay_patch(model: nn.Module) -> None:
    """Register MoE blocks and (for EP=1) wrap their forwards through
    ``_run_router()`` so that routing replay works on the HF native path.

    When EP > 1, ``expert_parallel.py`` already patches the forward through
    ``_run_router()`` — only registration is needed.
    """
    if getattr(model, '_rr_patched', False):
        return

    blocks = _find_moe_blocks_with_names(model)
    if not blocks:
        return

    # Walk *model*, find every MoE block and register it in ``_registry``.
    # Safe to call multiple times — already-registered blocks are skipped.
    for name, _ in blocks:
        if name not in _registry:
            _registry[name] = _RouterReplayState()

    # Determine if EP patches are already in place.
    # When EP > 1 the block forward has already been replaced by
    # patch_forward(); we only need to ensure replay_state is wired in.
    # When EP = 1 the original HF forward is intact — wrap it.

    # Check whether the first block's forward has already been EP-patched
    _, first_block = blocks[0]
    if _is_ep_patched(first_block):
        logger.debug('EP patches detected — routing replay piggy-backs on '
                     'patch_forward() replay_state wiring.')
        return

    # EP = 1: wrap each MoE block forward through _run_router()
    logger.info('Applying FSDP routing replay patch (EP=1 mode).')
    _wrap_all_moe_blocks(blocks)
    model._rr_patched = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_ep_patched(block: nn.Module) -> bool:
    """Return True if *block* has already been patched by expert_parallel."""
    return getattr(block, '_ep_patched', False)


def _find_moe_blocks_with_names(model: nn.Module) -> list[tuple[str, nn.Module]]:
    from .expert_parallel import find_moe_blocks_with_names

    # Strip PEFT wrapper so block names match those used by EP patch_forward
    unwrap = model.get_base_model() if hasattr(model, 'get_base_model') else model
    blocks = list(find_moe_blocks_with_names(unwrap))
    return blocks


def _wrap_all_moe_blocks(blocks: list[tuple[str, nn.Module]], ) -> None:
    """Replace each MoE block's forward with a wrapper that calls
    ``_run_router()`` instead of the original gate logic."""
    import types

    from .expert_parallel import (ExpertParallelConfig, _get_gate, _get_norm_topk_prob, _get_top_k,
                                  _maybe_run_shared_expert, _run_router)

    for name, block in blocks:
        gate = _get_gate(block)
        if gate is None:
            raise ValueError('MoE block must define gate/router module.')

        top_k = _get_top_k(block)
        if top_k is None:
            raise ValueError('MoE block must define top_k/num_experts_per_tok.')

        norm_topk_prob = _get_norm_topk_prob(block)

        original_forward = block.forward
        return_annotation = inspect.signature(original_forward).return_annotation
        returns_router_logits = return_annotation in (
            tuple,
            Tuple[torch.Tensor, torch.Tensor | None],
        )

        def _make_patched_forward(_name, _block, _gate, _top_k, _norm_topk_prob, _returns_router_logits):

            def patched_forward(self, hidden_states):
                orig_shape = hidden_states.shape
                if hidden_states.ndim == 3:
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hidden_states_2d = hidden_states.view(-1, hidden_dim)
                elif hidden_states.ndim == 2:
                    batch_size, sequence_length = 1, hidden_states.shape[0]
                    hidden_dim = hidden_states.shape[1]
                    hidden_states_2d = hidden_states
                else:
                    raise ValueError(f'Unsupported hidden_states ndim: {hidden_states.ndim}')

                replay_state = _registry.get(_name)
                router_logits, routing_weights, selected_experts = _run_router(
                    gate=_gate,
                    hidden_states=hidden_states_2d,
                    top_k=_top_k,
                    router_dtype=hidden_states_2d.dtype,
                    norm_topk_prob=_norm_topk_prob,
                    replay_state=replay_state,
                )

                if isinstance(self.experts, nn.ModuleList):
                    routed_output = torch.zeros_like(hidden_states_2d)
                    num_experts = len(self.experts)
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=num_experts).permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                    for expert_idx in expert_hit:
                        expert_layer = self.experts[expert_idx]
                        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                        current_state = hidden_states_2d[top_x]
                        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                        routed_output.index_add_(0, top_x, current_hidden_states.to(hidden_states_2d.dtype))
                else:
                    routed_output = self.experts(hidden_states_2d, selected_experts, routing_weights)

                shared_out = _maybe_run_shared_expert(_block, hidden_states_2d, ExpertParallelConfig())
                if shared_out is not None:
                    routed_output = routed_output + shared_out

                if len(orig_shape) == 3:
                    routed_output = routed_output.reshape(batch_size, sequence_length, hidden_dim)
                if _returns_router_logits:
                    return routed_output, router_logits
                return routed_output

            return patched_forward

        routed_fn = _make_patched_forward(name, block, gate, top_k, norm_topk_prob, returns_router_logits)
        block.forward = types.MethodType(routed_fn, block)
