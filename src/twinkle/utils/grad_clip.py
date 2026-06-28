# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gradient normalization and clipping utilities.

Architecture:
    normalize_and_clip_grad_norm (public entry point)
    ├── _normalize_grads           — divide all grads by num_tokens in-place
    ├── _ep_aware_clip_grad_norm   — EP two-phase reduce + unified clip
    │   └── _local_norm_stat       — per-rank norm statistic (foreach-accelerated)
    │       └── _collect_local_grads
    └── _standard_clip_grad_norm   — mixed DTensor/local tensor clip
        ├── _detect_grad_topology  — classify grads into DTensor/local/mixed
        ├── _resolve_reduce_device — pick device for all-reduce tensor
        ├── _compute_total_norm    — reduce local norms across ranks
        └── _apply_clip            — scale grads by clip coefficient
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Set, Tuple

from twinkle import Platform
from twinkle.utils import torch_util

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_and_clip_grad_norm(parameters: Iterable[torch.nn.Parameter],
                                 *,
                                 num_tokens: int,
                                 max_grad_norm: float,
                                 norm_type: float,
                                 group=None,
                                 ep_param_groups=None,
                                 ep_group=None,
                                 ep_fsdp_group=None) -> float:
    """Normalize gradients by *num_tokens*, then clip to *max_grad_norm*.

    Args:
        parameters: Trainable parameters whose grads will be normalized and clipped.
        num_tokens: Token count for gradient normalization (summed across DP ranks).
        max_grad_norm: Maximum allowed gradient norm after normalization.
        norm_type: Type of the norm (2.0 for L2, inf for max-norm).
        group: Process group for all-reduce (standard path, e.g. dp_group).
        ep_param_groups: If provided, ``{'ep': [...], 'non_ep': [...]}`` triggers
            EP-aware two-phase reduction.
        ep_group: Process group spanning EP ranks.
        ep_fsdp_group: Process group for FSDP shards within an EP partition.

    Returns:
        The total gradient norm (after normalization, before clipping).
    """
    parameters = list(parameters)
    grads = _normalize_grads(parameters, num_tokens)
    if not grads:
        return 0.0

    # EP-aware path: separate reduce for expert / non-expert params.
    if ep_param_groups is not None:
        return _ep_aware_clip_grad_norm(
            ep_param_groups=ep_param_groups,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            fsdp_group=group,
            ep_group=ep_group,
            ep_fsdp_group=ep_fsdp_group,
        )

    # Standard path: handles pure DTensor, pure local, and mixed cases.
    return _standard_clip_grad_norm(parameters, grads, max_grad_norm, norm_type, group)


# ---------------------------------------------------------------------------
# Gradient normalization
# ---------------------------------------------------------------------------


def _normalize_grads(parameters: list[torch.nn.Parameter], num_tokens: int) -> list[torch.Tensor]:
    """Divide every non-None gradient by *num_tokens* in-place and return the grad list."""
    if num_tokens <= 0:
        num_tokens = 1
    grads = []
    for param in parameters:
        if param.grad is None:
            continue
        param.grad.div_(num_tokens)
        grads.append(param.grad)
    return grads


# ---------------------------------------------------------------------------
# Standard (non-EP) clip path
# ---------------------------------------------------------------------------


def _standard_clip_grad_norm(
    parameters: list[torch.nn.Parameter],
    grads: list[torch.Tensor],
    max_grad_norm: float,
    norm_type: float,
    group,
) -> float:
    """Clip grads that may be a mix of DTensor and local Tensor."""
    import torch

    topology = _detect_grad_topology(grads)
    can_use_builtin = (not topology.has_mixed_mesh and (topology.all_dtensor or (topology.all_local and group is None)))

    if can_use_builtin:
        # PyTorch built-in handles DTensor reduce via mesh ops and works for
        # single-rank / DDP (grads pre-synced in backward).
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, norm_type=norm_type)
        grad_norm = torch_util.to_local_tensor(grad_norm)
        return float(grad_norm.item())

    # Manual path for mixed DTensor+local or explicit group reduce.
    norm_type = float(norm_type)
    if norm_type not in (2.0, float('inf')):
        raise ValueError('Mixed DTensor/Tensor clip_grad_norm only supports norm_type=2 or inf.')

    reduce_device = _resolve_reduce_device(grads)
    # Mixed meshes cannot be reduced via DTensor propagation; fall back to world group.
    reduce_group = None if topology.has_mixed_mesh else group
    total_norm = _compute_total_norm(grads, norm_type, reduce_device, reduce_group)
    _apply_clip(grads, max_grad_norm, total_norm)
    return total_norm


class _GradTopology:
    """Lightweight container describing the DTensor/local composition of grads."""
    __slots__ = ('all_dtensor', 'all_local', 'has_mixed_mesh')

    def __init__(self, all_dtensor: bool, all_local: bool, has_mixed_mesh: bool):
        self.all_dtensor = all_dtensor
        self.all_local = all_local
        self.has_mixed_mesh = has_mixed_mesh


def _detect_grad_topology(grads: list[torch.Tensor]) -> _GradTopology:
    """Classify gradients as pure-DTensor, pure-local, or mixed."""
    has_dtensor = False
    has_local = False
    mesh_keys: set = set()

    for grad in grads:
        if hasattr(grad, 'to_local'):
            has_dtensor = True
            mesh = getattr(grad, 'device_mesh', None)
            if mesh is None:
                mesh_keys.add('dtensor:unknown')
            else:
                try:
                    key = (tuple(mesh.mesh.flatten().tolist()), tuple(mesh.mesh_dim_names or ()))
                except Exception:
                    key = repr(mesh)
                mesh_keys.add(key)
        else:
            has_local = True

    return _GradTopology(
        all_dtensor=has_dtensor and not has_local,
        all_local=has_local and not has_dtensor,
        has_mixed_mesh=len(mesh_keys) > 1,
    )


def _resolve_reduce_device(grads: list[torch.Tensor]):
    """Pick the device to host the all-reduce scalar (prefer accelerator)."""
    import torch
    import torch.distributed as dist

    for grad in grads:
        local = grad.to_local() if hasattr(grad, 'to_local') else grad
        if local.is_cuda or getattr(local, 'is_npu', False):
            return local.device

    backend = dist.get_backend() if dist.is_initialized() else None
    if backend in ('nccl', 'hccl'):
        return torch.device(Platform.get_local_device())
    return torch.device('cpu')


def _compute_total_norm(
    grads: list[torch.Tensor],
    norm_type: float,
    device,
    group,
) -> float:
    """Compute the total gradient norm with cross-rank all-reduce."""
    import torch
    import torch.distributed as dist

    def _to_local(g):
        return g.to_local() if hasattr(g, 'to_local') else g

    if norm_type == float('inf'):
        local_val = 0.0
        for grad in grads:
            local_grad = _to_local(grad)
            if local_grad.numel() == 0:
                continue
            local_val = max(local_val, local_grad.detach().abs().max().item())
        total_tensor = torch.tensor(local_val, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX, group=group)
    else:
        local_sq = 0.0
        for grad in grads:
            local_grad = _to_local(grad)
            if local_grad.numel() == 0:
                continue
            local_sq += local_grad.detach().float().pow(2).sum().item()
        total_tensor = torch.tensor(local_sq, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM, group=group)
        total_tensor = total_tensor.sqrt()

    return float(total_tensor.item())


def _apply_clip(grads: list[torch.Tensor], max_grad_norm: float, total_norm: float) -> None:
    """Scale gradients in-place if total_norm exceeds max_grad_norm."""
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)


# ---------------------------------------------------------------------------
# EP-aware clip path
# ---------------------------------------------------------------------------


def _ep_aware_clip_grad_norm(
    *,
    ep_param_groups,
    max_grad_norm: float,
    norm_type: float,
    fsdp_group=None,
    ep_group=None,
    ep_fsdp_group=None,
) -> float:
    """EP-aware gradient clipping with two-phase reduction.

    Reduction strategy:
        - non-EP params: all-reduce over *fsdp_group*
        - EP params: all-reduce over *ep_fsdp_group*, then *ep_group*
    After obtaining the unified total norm, applies clipping to both groups.
    """
    import math
    import torch

    ep_params = [p for p in ep_param_groups.get('ep', []) if p.grad is not None and p.requires_grad]
    non_ep_params = [p for p in ep_param_groups.get('non_ep', []) if p.grad is not None and p.requires_grad]
    norm_type = float(norm_type)

    with torch.no_grad():
        non_ep_val = _local_norm_stat(non_ep_params, norm_type)
        ep_val = _local_norm_stat(ep_params, norm_type)
        _reduce_norm_stat(non_ep_val, norm_type, fsdp_group)
        _reduce_norm_stat(ep_val, norm_type, ep_fsdp_group)
        _reduce_norm_stat(ep_val, norm_type, ep_group)

        # Combine
        if math.isinf(norm_type):
            total_norm = torch.maximum(non_ep_val, ep_val)
        else:
            total_norm = (non_ep_val + ep_val)**(1.0 / norm_type)

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_grad_norm, total_norm, foreach=True)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_grad_norm, total_norm, foreach=True)
    return float(total_norm.item())


def _reduce_norm_stat(val, norm_type: float, group) -> None:
    """All-reduce a norm statistic tensor over the given group (no-op if group is None)."""
    if group is None:
        return
    import math
    import torch.distributed as dist
    op = dist.ReduceOp.MAX if math.isinf(norm_type) else dist.ReduceOp.SUM
    dist.all_reduce(val, op=op, group=group)


# ---------------------------------------------------------------------------
# Local norm computation (foreach-accelerated)
# ---------------------------------------------------------------------------


def _local_norm_stat(params, norm_type: float):
    """Compute the local (single-rank) norm statistic.

    Returns:
        A scalar tensor on accelerator: sum-of-p-th-powers (finite p) or max-abs (inf).
    """
    import math

    grads_local, default_device = _collect_local_grads(params)

    if math.isinf(norm_type):
        return _local_inf_norm(grads_local, default_device)
    return _local_p_norm_stat(grads_local, norm_type, default_device)


def _collect_local_grads(params) -> tuple[list, torch.device]:
    """Extract local fp32 grad tensors and determine the compute device."""
    import torch
    from torch.distributed._tensor import DTensor

    grads_local = []
    default_device = None
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        if default_device is None and (g.is_cuda or getattr(g, 'is_npu', False)):
            default_device = g.device
        grads_local.append(g.detach().to(torch.float32))

    if default_device is None:
        default_device = torch.device(Platform.get_local_device())

    return grads_local, default_device


def _local_inf_norm(grads_local: list, device) -> torch.Tensor:
    """Compute local max-abs norm."""
    import torch
    val = torch.tensor(0.0, device=device, dtype=torch.float32)
    for g in grads_local:
        if g.numel() == 0:
            continue
        val = torch.maximum(val, g.abs().max())
    return val


def _local_p_norm_stat(grads_local: list, norm_type: float, device) -> torch.Tensor:
    """Compute sum of p-th powers of per-grad norms (foreach-accelerated)."""
    import torch

    p = float(norm_type)
    val = torch.tensor(0.0, device=device, dtype=torch.float32)
    non_empty = [g for g in grads_local if g.numel() > 0]
    if not non_empty:
        return val

    # Try vectorized foreach path (private PyTorch util, may be absent in future).
    try:
        from torch.utils._foreach_utils import (_device_has_foreach_support, _group_tensors_by_device_and_dtype,
                                                _has_foreach_support)
    except ImportError:
        return _local_p_norm_stat_scalar(non_empty, p, val)

    grouped = _group_tensors_by_device_and_dtype([non_empty])
    for (dev, _), ([device_grads], _) in grouped.items():
        if _has_foreach_support(device_grads, dev) or _device_has_foreach_support(dev):
            # NOTE: _foreach_pow_ is in-place and returns None (PyTorch convention);
            # we must keep the intermediate list reference.
            norms = torch._foreach_norm(device_grads, p)
            torch._foreach_pow_(norms, p)
            val += torch.sum(torch.stack(norms)).to(device)
        else:
            for g in device_grads:
                val += torch.norm(g, p=p).pow(p).to(device)
    return val


def _local_p_norm_stat_scalar(grads: list, p: float, val) -> torch.Tensor:
    """Scalar fallback for p-norm stat when foreach utilities are unavailable."""
    import torch
    device = val.device
    for g in grads:
        val += torch.norm(g, p=p).pow(p).to(device)
    return val
