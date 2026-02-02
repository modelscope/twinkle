# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Any, Optional, Literal, Set

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

from twinkle.utils import DeviceMesh


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

        fsdp_mesh = _build_fsdp_mesh(self.device_mesh)
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


def _build_fsdp_mesh(device_mesh: DeviceMesh) -> Optional[TorchDeviceMesh]:
    if device_mesh is None or device_mesh.mesh_dim_names is None:
        return None
    flat_mesh = device_mesh.mesh.flatten()
    if flat_mesh.size <= 1:
        return None
    return TorchDeviceMesh(device_mesh.device_type, flat_mesh, mesh_dim_names=("fsdp",))


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
