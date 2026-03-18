# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from twinkle import Platform
from twinkle.utils import torch_util

if TYPE_CHECKING:
    import torch


def normalize_and_clip_grad_norm(parameters: Iterable[torch.nn.Parameter],
                                 *,
                                 num_tokens: int,
                                 max_grad_norm: float,
                                 norm_type: float,
                                 group=None,
                                 ep_param_groups=None,
                                 ep_group=None,
                                 ep_fsdp_group=None) -> float:
    """Normalize gradients by num_tokens, then clip by max_grad_norm.

    If ep_param_groups is provided, uses EP-aware two-phase reduction:
    - non-EP params: all-reduce over group (fsdp_group)
    - EP params: all-reduce over ep_fsdp_group, then ep_group
    """
    import torch
    import torch.distributed as dist
    parameters = list(parameters)
    if num_tokens <= 0:
        num_tokens = 1

    grads = []
    for param in parameters:
        if param.grad is None:
            continue
        param.grad.div_(num_tokens)
        grads.append(param.grad)

    if not grads:
        return 0.0

    # EP-aware path
    if ep_param_groups is not None:
        return _ep_aware_clip_grad_norm(
            ep_param_groups=ep_param_groups,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            fsdp_group=group,
            ep_group=ep_group,
            ep_fsdp_group=ep_fsdp_group,
        )

    # Standard path (backward compatible)
    has_dtensor_grad = any(hasattr(grad, 'to_local') for grad in grads)
    has_local_tensor_grad = any(not hasattr(grad, 'to_local') for grad in grads)
    dtensor_mesh_keys = set()
    for grad in grads:
        if not hasattr(grad, 'to_local'):
            continue
        mesh = getattr(grad, 'device_mesh', None)
        if mesh is None:
            dtensor_mesh_keys.add('dtensor:unknown')
            continue
        try:
            mesh_key = (tuple(mesh.mesh.flatten().tolist()), tuple(mesh.mesh_dim_names or ()))
        except Exception:
            mesh_key = repr(mesh)
        dtensor_mesh_keys.add(mesh_key)

    has_mixed_dtensor_mesh = len(dtensor_mesh_keys) > 1

    if not (has_dtensor_grad and has_local_tensor_grad) and not has_mixed_dtensor_mesh:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_grad_norm,
            norm_type=norm_type,
        )
        grad_norm = torch_util.to_local_tensor(grad_norm)
        return float(grad_norm.item())

    norm_type = float(norm_type)
    if norm_type not in (2.0, float('inf')):
        raise ValueError('Mixed DTensor/Tensor clip_grad_norm only supports norm_type=2 or inf.')

    def _local_grad(grad: torch.Tensor) -> torch.Tensor:
        if hasattr(grad, 'to_local'):
            return grad.to_local()
        return grad

    reduce_device = None
    for grad in grads:
        local_grad = _local_grad(grad)
        if local_grad.is_cuda or getattr(local_grad, 'is_npu', False):
            reduce_device = local_grad.device
            break
    if reduce_device is None:
        backend = dist.get_backend() if dist.is_initialized() else None
        if backend in ('nccl', 'hccl'):
            reduce_device = torch.device(Platform.get_local_device())
        else:
            reduce_device = torch.device('cpu')
    reduce_group = group
    if has_mixed_dtensor_mesh:
        # Different DTensor meshes cannot be reduced by DTensor op propagation (e.g. aten.stack).
        # Fall back to world reduction over local shards.
        reduce_group = None

    if norm_type == float('inf'):
        local_norm = 0.0
        for grad in grads:
            local_grad = _local_grad(grad)
            if local_grad.numel() == 0:
                continue
            local_norm = max(local_norm, local_grad.detach().abs().max().item())
        total_norm_tensor = torch.tensor(local_norm, device=reduce_device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(total_norm_tensor, op=dist.ReduceOp.MAX, group=reduce_group)
        total_norm = float(total_norm_tensor.item())
    else:
        local_sq = 0.0
        for grad in grads:
            local_grad = _local_grad(grad)
            if local_grad.numel() == 0:
                continue
            local_sq += local_grad.detach().float().pow(2).sum().item()
        total_sq_tensor = torch.tensor(local_sq, device=reduce_device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(total_sq_tensor, op=dist.ReduceOp.SUM, group=reduce_group)
        total_norm = float(total_sq_tensor.sqrt().item())

    clip_coef = float(max_grad_norm) / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return total_norm


def _ep_aware_clip_grad_norm(
    *,
    ep_param_groups,
    max_grad_norm: float,
    norm_type: float,
    fsdp_group=None,
    ep_group=None,
    ep_fsdp_group=None,
) -> float:
    """EP-aware gradient clipping.

    - non-EP params: all-reduce over fsdp_group
    - EP params: all-reduce over ep_fsdp_group, then ep_group
    - Unified clip coefficient applied to both groups via clip_grads_with_norm_
    """
    import math
    import torch
    import torch.distributed as dist

    ep_params = [p for p in ep_param_groups.get('ep', []) if p.grad is not None]
    non_ep_params = [p for p in ep_param_groups.get('non_ep', []) if p.grad is not None]

    norm_type = float(norm_type)

    with torch.no_grad():
        # non-EP: reduce over fsdp_group
        non_ep_val = _local_norm_stat(non_ep_params, norm_type)
        if fsdp_group is not None:
            op = dist.ReduceOp.MAX if math.isinf(norm_type) else dist.ReduceOp.SUM
            dist.all_reduce(non_ep_val, op=op, group=fsdp_group)

        # EP: reduce over ep_fsdp_group, then ep_group
        ep_val = _local_norm_stat(ep_params, norm_type)
        if ep_fsdp_group is not None:
            op = dist.ReduceOp.MAX if math.isinf(norm_type) else dist.ReduceOp.SUM
            dist.all_reduce(ep_val, op=op, group=ep_fsdp_group)
        if ep_group is not None:
            op = dist.ReduceOp.MAX if math.isinf(norm_type) else dist.ReduceOp.SUM
            dist.all_reduce(ep_val, op=op, group=ep_group)

        # Combine into total_norm tensor
        if math.isinf(norm_type):
            total_norm = torch.maximum(non_ep_val, ep_val)
        else:
            total_norm = (non_ep_val + ep_val)**(1.0 / norm_type)

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_grad_norm, total_norm, foreach=True)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_grad_norm, total_norm, foreach=True)

    return float(total_norm.item())


def _local_norm_stat(params, norm_type: float):
    """Compute local norm statistic: sum of p-th powers (finite p) or max (inf).

    Uses torch._foreach_* batch kernels for finite p to reduce kernel launch overhead.
    """
    import math
    import torch
    from torch.distributed._tensor import DTensor
    from torch.utils._foreach_utils import (_device_has_foreach_support, _group_tensors_by_device_and_dtype,
                                            _has_foreach_support)

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

    if math.isinf(norm_type):
        val = torch.tensor(0.0, device=default_device, dtype=torch.float32)
        for g in grads_local:
            if g.numel() == 0:
                continue
            val = torch.maximum(val, g.abs().max())
        return val

    p = float(norm_type)
    val = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    if not grads_local:
        return val
    non_empty = [g for g in grads_local if g.numel() > 0]
    if not non_empty:
        return val
    grouped = _group_tensors_by_device_and_dtype([non_empty])
    for (device, _), ([device_grads], _) in grouped.items():
        if _has_foreach_support(device_grads, device) or _device_has_foreach_support(device):
            # Batch: compute ||g||_p for each grad, raise to p-th power, then sum
            out = torch._foreach_pow_(torch._foreach_norm(device_grads, p), p)
            val += torch.sum(torch.stack(out)).to(default_device)
        else:
            for g in device_grads:
                val += (torch.norm(g, p=p)**p).to(default_device)
    return val
