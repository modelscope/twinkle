# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic per-expert Python loop backend for the EP experts interface.

This is the always-eligible fallback: it runs on any platform (pure PyTorch)
and is registered last, so the dispatcher can never run out of backends.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import EpExpertsGmm


class LoopEpExpertsGmm(EpExpertsGmm):
    """Per-expert Python loop over the permuted token buffer (generic fallback)."""

    name = 'per-expert loop'
    fallback = True

    def ineligible_reason(self, experts_mod: nn.Module) -> str | None:
        # The loop is the reference implementation: always eligible.
        return None

    def forward(
        self,
        experts_mod: nn.Module,
        permuted_tokens: torch.Tensor,
        num_global_sum_tokens_per_local_expert: torch.Tensor,
        experts_per_rank: int,
    ) -> torch.Tensor:
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

            gate_up = experts_mod.gate_up_proj[i]
            down = experts_mod.down_proj[i]
            compute_dtype = gate_up.dtype
            if expert_in.dtype != compute_dtype:
                expert_in = expert_in.to(compute_dtype)
            gate_up_out = F.linear(expert_in, gate_up)
            if hasattr(experts_mod, '_apply_gate'):
                out = experts_mod._apply_gate(gate_up_out)
            else:
                gate, up = gate_up_out.chunk(2, dim=-1)
                out = experts_mod.act_fn(gate) * up
            out = F.linear(out, down)

            if out.dtype != input_dtype:
                out = out.to(input_dtype)
            output_chunks.append(out)

        return torch.cat(
            output_chunks, dim=0) if output_chunks else permuted_tokens.new_empty(0, permuted_tokens.size(-1))
