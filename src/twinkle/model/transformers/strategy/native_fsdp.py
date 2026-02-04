# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal, Set, Callable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

from twinkle.utils import DeviceMesh, Platform


class NativeFSDPStrategy:
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

        dense_fsdp_mesh = _build_dense_fsdp_mesh(self.device_mesh)
        if dense_fsdp_mesh is not None:
            _ensure_moe_patched_if_needed(model, self.device_mesh)
            _place_ep_experts_on_local_device(model, self.device_mesh)
            mp_policy = _build_mp_policy(self.mixed_precision)
            reshard_after_forward = self.fsdp_config.get("reshard_after_forward", True)
            efsdp_config = _resolve_efsdp_config(model)
            if efsdp_config.enabled:
                expert_fsdp_mesh, expert_ranks = _build_expert_fsdp_mesh(
                    self.device_mesh, efsdp_config.mesh_dim)
                _shard_expert_modules_with_efsdp(
                    model,
                    mesh=expert_fsdp_mesh,
                    group_ranks=expert_ranks,
                    reshard_after_forward=reshard_after_forward,
                    mp_policy=mp_policy,
                    default_shard_dim=efsdp_config.shard_dim,
                )
            ignored_params = _collect_expert_params(model)

            _maybe_shard_layers(
                model,
                mesh=dense_fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                ignored_params=ignored_params,
            )
            _fully_shard_module(
                model,
                mesh=dense_fsdp_mesh,
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


def _build_dense_fsdp_mesh(device_mesh: DeviceMesh) -> Optional[TorchDeviceMesh]:
    if device_mesh is None or device_mesh.mesh_dim_names is None:
        return None
    flat_mesh = device_mesh.mesh.flatten()
    if flat_mesh.size <= 1:
        return None
    return TorchDeviceMesh(device_mesh.device_type, flat_mesh, mesh_dim_names=("fsdp",))


@dataclass
class _EfsdpConfig:
    enabled: bool = False
    shard_dim: int = 1
    mesh_dim: str = "dp"


def _resolve_efsdp_config(model: nn.Module) -> _EfsdpConfig:
    config = _EfsdpConfig()
    for module in model.modules():
        if not getattr(module, "_ep_patched", False):
            continue
        if not getattr(module, "_ep_efsdp_enabled", False):
            continue
        shard_dim = int(getattr(module, "_ep_efsdp_shard_dim", 1))
        mesh_dim = str(getattr(module, "_ep_efsdp_mesh_dim", "dp"))
        if not config.enabled:
            config.enabled = True
            config.shard_dim = shard_dim
            config.mesh_dim = mesh_dim
            continue
        if shard_dim != config.shard_dim or mesh_dim != config.mesh_dim:
            raise ValueError(
                "Inconsistent eFSDP config detected across MoE blocks. "
                f"Expected shard_dim={config.shard_dim}, mesh_dim={config.mesh_dim}; "
                f"got shard_dim={shard_dim}, mesh_dim={mesh_dim}."
            )
    return config


def _build_expert_fsdp_mesh(device_mesh: DeviceMesh,
                            mesh_dim: str) -> Tuple[Optional[TorchDeviceMesh], List[int]]:
    if device_mesh is None or not device_mesh.has_dim(mesh_dim):
        raise ValueError(
            f"eFSDP mesh_dim='{mesh_dim}' is not found in device mesh {device_mesh.mesh_dim_names}."
        )
    ranks = sorted(device_mesh.get_ranks_in_dim(mesh_dim))
    if len(ranks) <= 1:
        return None, ranks
    mesh = TorchDeviceMesh(
        device_mesh.device_type,
        np.array(ranks),
        mesh_dim_names=("efsdp",),
    )
    return mesh, ranks


def _collect_expert_params(model: nn.Module) -> Optional[Set[nn.Parameter]]:
    ignored: Set[nn.Parameter] = set()
    ep_patched = False
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is not None and getattr(module, "_ep_patched", False):
            ep_patched = True
            if isinstance(experts, nn.ModuleList):
                for expert in experts:
                    ignored.update(expert.parameters())
            else:
                ignored.update(experts.parameters())

        if getattr(module, "_ep_ignore_shared_experts", False) and getattr(module, "_ep_patched", False):
            ep_patched = True
            shared = getattr(module, "shared_expert", None)
            if shared is not None:
                ignored.update(shared.parameters())

    if not ep_patched:
        return None
    return ignored or None


def _place_ep_experts_on_local_device(model: nn.Module, device_mesh: DeviceMesh) -> None:
    ep_world_size = device_mesh.ep_world_size or 1
    if ep_world_size <= 1:
        return
    local_device = torch.device(Platform.get_local_device())
    for module in model.modules():
        if not getattr(module, "_ep_patched", False):
            continue
        experts = getattr(module, "experts", None)
        if experts is not None:
            experts.to(local_device)
        if getattr(module, "_ep_ignore_shared_experts", False):
            shared = getattr(module, "shared_expert", None)
            if shared is not None:
                shared.to(local_device)


def _ensure_moe_patched_if_needed(model: nn.Module, device_mesh: DeviceMesh) -> None:
    ep_world_size = device_mesh.ep_world_size or 1
    if ep_world_size <= 1:
        return
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if isinstance(experts, nn.ModuleList) and not getattr(module, "_ep_patched", False):
            raise RuntimeError(
                "Found MoE experts but expert parallel is not applied. "
                "Call apply_expert_parallel(model, device_mesh, config) before wrapping with FSDP2."
            )


def _fully_shard_module(module: nn.Module,
                        *,
                        mesh: TorchDeviceMesh,
                        reshard_after_forward: Optional[bool],
                        mp_policy: MixedPrecisionPolicy,
                        ignored_params: Optional[Set[nn.Parameter]] = None,
                        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[object]]] = None
                        ) -> None:
    kwargs = {
        "mesh": mesh,
        "reshard_after_forward": reshard_after_forward,
        "mp_policy": mp_policy,
    }
    if ignored_params is not None:
        kwargs["ignored_params"] = ignored_params
    if shard_placement_fn is not None:
        kwargs["shard_placement_fn"] = shard_placement_fn
    try:
        fully_shard(module, **kwargs)
    except TypeError as exc:
        if shard_placement_fn is not None and "shard_placement_fn" in str(exc):
            raise RuntimeError(
                "Current torch fully_shard does not support shard_placement_fn. "
                "eFSDP requires torch>=2.6."
            ) from exc
        raise


def _shard_expert_modules_with_efsdp(model: nn.Module,
                                     *,
                                     mesh: Optional[TorchDeviceMesh],
                                     group_ranks: List[int],
                                     reshard_after_forward: Optional[bool],
                                     mp_policy: MixedPrecisionPolicy,
                                     default_shard_dim: int) -> None:
    if mesh is None:
        return

    expert_modules: List[nn.Module] = []
    param_to_shard_dim: Dict[int, int] = {}
    for module in model.modules():
        if not getattr(module, "_ep_patched", False):
            continue
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        module._ep_efsdp_ranks = tuple(group_ranks)
        if experts in expert_modules:
            continue
        expert_modules.append(experts)
        layout = getattr(module, "_ep_expert_param_layout", {})
        for param_name, param in experts.named_parameters():
            layout_info = layout.get(param_name, {})
            shard_dim = int(layout_info.get("efsdp_shard_dim", default_shard_dim))
            if param.ndim <= shard_dim:
                raise ValueError(
                    f"Invalid eFSDP shard dim {shard_dim} for expert param '{param_name}' "
                    f"with shape {tuple(param.shape)}."
                )
            param_to_shard_dim[id(param)] = shard_dim

    def _placement_fn(param: nn.Parameter):
        shard_dim = param_to_shard_dim.get(id(param))
        if shard_dim is None:
            return None
        from torch.distributed.tensor import Shard
        return Shard(shard_dim)

    for experts in expert_modules:
        _fully_shard_module(
            experts,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            shard_placement_fn=_placement_fn,
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
        _fully_shard_module(
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
    name_to_param = dict(model.named_parameters())
    ep_patched = any(getattr(module, "_ep_patched", False) for module in model.modules())
    if len(optimizer.param_groups) != 1:
        for group in optimizer.param_groups:
            if "param_names" not in group:
                raise RuntimeError(
                    "NativeFSDPStrategy cannot rebind optimizer param_groups without param_names. "
                    "Create the optimizer after wrapping, or include param_names in each group."
                )
            new_params = []
            for name in group["param_names"]:
                if name not in name_to_param:
                    if ep_patched and ".experts." in name:
                        continue
                    raise RuntimeError(
                        f"NativeFSDPStrategy could not find parameter '{name}' when rebinding optimizer."
                    )
                new_params.append(name_to_param[name])
            group["params"] = new_params
        return optimizer
    optimizer.param_groups[0]["params"] = list(model.parameters())
    return optimizer
