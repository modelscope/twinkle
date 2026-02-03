# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Iterable, TYPE_CHECKING
from twinkle import Platform
from twinkle.utils import torch_util
if TYPE_CHECKING:
    import torch


def normalize_and_clip_grad_norm(parameters: Iterable['torch.nn.Parameter'],
                                 *,
                                 num_tokens: int,
                                 max_grad_norm: float,
                                 norm_type: float,
                                 group=None) -> float:
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

    has_dtensor_grad = any(hasattr(grad, "to_local") for grad in grads)
    has_local_tensor_grad = any(not hasattr(grad, "to_local") for grad in grads)
    if not (has_dtensor_grad and has_local_tensor_grad):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_grad_norm,
            norm_type=norm_type,
        )
        grad_norm = torch_util.to_local_tensor(grad_norm)
        return float(grad_norm.item())

    norm_type = float(norm_type)
    if norm_type not in (2.0, float("inf")):
        raise ValueError("Mixed DTensor/Tensor clip_grad_norm only supports norm_type=2 or inf.")

    def _local_grad(grad: torch.Tensor) -> torch.Tensor:
        if hasattr(grad, "to_local"):
            return grad.to_local()
        return grad

    reduce_device = None
    for grad in grads:
        local_grad = _local_grad(grad)
        if local_grad.is_cuda:
            reduce_device = local_grad.device
            break
    if reduce_device is None:
        backend = dist.get_backend() if dist.is_initialized() else None
        if backend == "nccl":
            reduce_device = torch.device(Platform.get_local_device())
        else:
            reduce_device = torch.device("cpu")

    if norm_type == float("inf"):
        local_norm = 0.0
        for grad in grads:
            local_grad = _local_grad(grad)
            if local_grad.numel() == 0:
                continue
            local_norm = max(local_norm, local_grad.detach().abs().max().item())
        total_norm_tensor = torch.tensor(local_norm, device=reduce_device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(total_norm_tensor, op=dist.ReduceOp.MAX, group=group)
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
            dist.all_reduce(total_sq_tensor, op=dist.ReduceOp.SUM, group=group)
        total_norm = float(total_sq_tensor.sqrt().item())

    clip_coef = float(max_grad_norm) / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return total_norm
