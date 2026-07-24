# Copyright (c) ModelScope Contributors. All rights reserved.
"""NPU backend for the EP experts grouped-matmul interface."""
from __future__ import annotations

import torch
from torch import nn

from . import EpExpertsGmm


class NpuEpExpertsGmm(EpExpertsGmm):
    """Batched EP experts compute via ``torch_npu.npu_grouped_matmul``."""

    name = 'NPU'

    def ineligible_reason(self, experts_mod: nn.Module) -> str | None:
        """Check whether the tensor-experts EP compute can use NPU grouped matmul.

        Falls back to the per-expert Python loop when:
        - torch_npu / NPU is unavailable;
        - there is no way to compute the gated activation: no ``_apply_gate``
          hook and a non-SiLU ``act_fn`` (``npu_swiglu`` == silu(gate) * up);
        - the packed weights are not plain 3D floating tensors (e.g. DTensor
          that was not unsharded, quantized weights, etc.).

        Note on ``_apply_gate``: transformers injects a default implementation
        (``_default_apply_gate``: chunk + act_fn(gate) * up) onto many MoE
        expert classes. With SiLU activation that is exactly ``npu_swiglu``, so
        it does not block the fused path. A *custom* gate hook is still
        supported — it is called eagerly on the grouped-matmul output.
        """
        apply_gate = getattr(experts_mod, '_apply_gate', None)
        is_default_gate = getattr(apply_gate, '__name__', None) == '_default_apply_gate'
        if apply_gate is None or is_default_gate:
            # Activation will be computed by npu_swiglu == silu(gate) * up.
            act_fn = getattr(experts_mod, 'act_fn', None)
            if act_fn is None:
                return 'no act_fn attribute'
            # nn.SiLU and transformers' SiLUActivation both compute F.silu
            # exactly; either is equivalent to the silu inside npu_swiglu.
            if act_fn.__class__.__name__ not in ('SiLU', 'SiLUActivation'):
                return f'act_fn is {act_fn.__class__.__name__}, not SiLU'
        # else: custom gate hook is called eagerly in the GMM path — allowed.
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return 'torch_npu not importable'
        if not torch.npu.is_available():
            return 'NPU not available'
        for name in ('gate_up_proj', 'down_proj'):
            weight = getattr(experts_mod, name, None)
            # Note: accept plain tensors and nn.Parameter. Under FSDP2 the
            # pre-forward hook has already unsharded params by the time
            # ep_forward runs; under PEFT target_parameters the parametrized
            # property returns a merged plain tensor. Anything else (DTensor,
            # quantized types) falls back to the loop.
            if not isinstance(weight, torch.Tensor) or weight.__class__.__name__ == 'DTensor':
                return f'{name} is {type(weight).__name__}, not a plain tensor'
            if weight.ndim != 3 or not weight.dtype.is_floating_point:
                return f'{name} has shape {tuple(weight.shape)} dtype {weight.dtype}'
        return None

    def forward(
        self,
        experts_mod: nn.Module,
        permuted_tokens: torch.Tensor,
        num_global_sum_tokens_per_local_expert: torch.Tensor,
        experts_per_rank: int,
    ) -> torch.Tensor:
        """Compute local experts with one grouped matmul instead of a Python loop.

        Mathematically equivalent to the per-expert loop: grouped matmul
        applies each expert's weight to its contiguous token chunk, and
        ``npu_swiglu(x) == silu(gate) * up`` for gate/up = x.chunk(2, -1).
        The group list stays on device, eliminating the per-expert ``.item()``
        host synchronizations of the loop implementation.
        """
        import torch_npu

        from twinkle.kernel.npu_impls.moe import GmmFunction, _get_cached_expert_weights

        input_dtype = permuted_tokens.dtype
        hidden_dim = permuted_tokens.size(-1)
        # counts arrives as a CPU tensor from preprocess(); grouped matmul needs
        # the group list on device.
        group_list = num_global_sum_tokens_per_local_expert.to(device=permuted_tokens.device, dtype=torch.int64)
        gate_up_weight, down_weight = _get_cached_expert_weights(experts_mod, input_dtype, hidden_dim)

        expert_in = permuted_tokens
        if expert_in.dtype != gate_up_weight.dtype:
            expert_in = expert_in.to(gate_up_weight.dtype)
        intermediate = GmmFunction.apply(expert_in, group_list, gate_up_weight)
        apply_gate = getattr(experts_mod, '_apply_gate', None)
        if apply_gate is not None and getattr(apply_gate, '__name__', '') != '_default_apply_gate':
            # Custom gate hook: honor it eagerly instead of the fused swiglu.
            activated = apply_gate(intermediate)
        else:
            # No hook, or transformers' default gate: silu(gate) * up == npu_swiglu.
            activated = torch_npu.npu_swiglu(intermediate, dim=-1)
        out = GmmFunction.apply(activated, group_list, down_weight)
        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        return out
