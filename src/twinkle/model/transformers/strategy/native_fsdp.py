# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, Optional, Literal, Set, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

from .base import TrainStrategy
from twinkle.utils import DeviceMesh


class NativeFSDPStrategy(TrainStrategy):
    """FSDP2 strategy with explicit process group control for EP compatibility."""

    def __init__(self,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 fsdp_config: Dict[str, Any] = None):
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.fsdp_config = fsdp_config or {}

    def wrap_model(self, model, optimizer=None):
        if self.device_mesh is None:
            return model, optimizer

        fsdp_mesh = _build_fsdp_mesh(self.device_mesh, ("dp", "fsdp"))
        if fsdp_mesh is not None:
            _ensure_moe_patched_if_needed(model, self.device_mesh)
            mp_policy = _build_mp_policy(self.mixed_precision)
            reshard_after_forward = self.fsdp_config.get("reshard_after_forward", True)
            ignored_params = _collect_expert_params(model)

            _maybe_shard_layers(
                model,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                ignored_params=ignored_params,
            )
            fully_shard(
                model,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                ignored_params=ignored_params,
            )

        if optimizer is not None:
            optimizer = _rebind_optimizer(optimizer, model)

        return model, optimizer

    def unwrap_model(self, model):
        return model


def _build_mp_policy(mixed_precision: str) -> MixedPrecisionPolicy:
    if mixed_precision == "bf16":
        dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        dtype = torch.float16
    else:
        return MixedPrecisionPolicy()
    return MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
        cast_forward_inputs=True,
    )


def _build_fsdp_mesh(device_mesh: DeviceMesh, dims: Tuple[str, ...]) -> Optional[TorchDeviceMesh]:
    if device_mesh is None or device_mesh.mesh_dim_names is None:
        return None

    dims_in_order = tuple(name for name in device_mesh.mesh_dim_names if name in dims)
    if not dims_in_order:
        return None

    coord = device_mesh._get_coord()
    indices = []
    for i, name in enumerate(device_mesh.mesh_dim_names):
        if name in dims_in_order:
            indices.append(slice(None))
        else:
            indices.append(coord[i])

    sub_mesh = device_mesh.mesh[tuple(indices)]
    if sub_mesh.size <= 1:
        return None
    return TorchDeviceMesh(device_mesh.device_type, sub_mesh, mesh_dim_names=dims_in_order)


def _collect_expert_params(model: nn.Module) -> Optional[Set[nn.Parameter]]:
    ignored: Set[nn.Parameter] = set()
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList):
            for expert in experts:
                ignored.update(expert.parameters())

        if getattr(module, "_ep_ignore_shared_experts", False):
            shared = getattr(module, "shared_expert", None)
            if shared is not None:
                ignored.update(shared.parameters())

    return ignored or None


def _ensure_moe_patched_if_needed(model: nn.Module, device_mesh: DeviceMesh) -> None:
    if device_mesh.ep_world_size <= 1:
        return
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList) and not getattr(module, "_ep_patched", False):
            raise RuntimeError(
                "Found MoE experts but expert parallel is not applied. "
                "Call apply_expert_parallel(model, device_mesh, config) before wrapping with FSDP2."
            )


def _maybe_shard_layers(model: nn.Module,
                        *,
                        mesh: TorchDeviceMesh,
                        reshard_after_forward: Optional[bool],
                        mp_policy: MixedPrecisionPolicy,
                        ignored_params: Optional[Set[nn.Parameter]]) -> None:
    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        return
    for layer in layers:
        fully_shard(
            layer,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            ignored_params=ignored_params,
        )


def _rebind_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module) -> torch.optim.Optimizer:
    if optimizer.state:
        raise RuntimeError(
            "Optimizer already has state. Create the optimizer after FSDP wrapping, "
            "or reinitialize it before training."
        )
    if len(optimizer.param_groups) != 1:
        raise RuntimeError(
            "NativeFSDPStrategy currently supports a single optimizer param_group. "
            "Please construct the optimizer after wrapping to preserve custom groups."
        )
    optimizer.param_groups[0]["params"] = list(model.parameters())
    return optimizer
