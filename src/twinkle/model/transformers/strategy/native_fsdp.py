# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed.fsdp import fully_shard
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Set

from twinkle.utils import DeviceMesh, Platform

if TYPE_CHECKING:
    from torch.distributed.fsdp import MixedPrecisionPolicy


class NativeFSDPStrategy:

    def __init__(self,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 fsdp_config: Dict[str, Any] = None,
                 enable_ep: bool = True,
                 ep_size: Optional[int] = None):
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.fsdp_config = fsdp_config or {}
        self.enable_ep = enable_ep
        self.ep_fsdp_device_mesh = self._build_ep_fsdp_device_mesh(ep_size) if enable_ep else None

    def _build_ep_fsdp_device_mesh(self, ep_size: Optional[int] = None) -> Optional[TorchDeviceMesh]:
        if self.device_mesh is None:
            return None
        ep_size = ep_size or self.device_mesh.ep_size or 1
        if ep_size <= 1:
            return None
        import numpy as np
        world_size = self.device_mesh.world_size
        ep_fsdp_size = self.device_mesh.ep_fsdp_size or (world_size // ep_size)
        ep_mesh = DeviceMesh(
            mesh=np.arange(world_size).reshape(ep_size, ep_fsdp_size),
            mesh_dim_names=('ep', 'ep_fsdp'),
            device_type=self.device_mesh.device_type,
        )
        return ep_mesh.to_torch_device_mesh()

    def wrap_model(self, model, optimizer=None):
        if self.device_mesh is None:
            return model, optimizer
        fsdp_mesh = _build_fsdp_mesh(self.device_mesh)
        if fsdp_mesh is not None:
            ep_enabled = (self.enable_ep and self.ep_fsdp_device_mesh is not None)
            if ep_enabled:
                _ensure_moe_patched_if_needed(model, self.ep_fsdp_device_mesh)
                _place_ep_experts_on_local_device(model, self.ep_fsdp_device_mesh)
            mp_policy = _build_mp_policy(self.mixed_precision)
            reshard_after_forward = self.fsdp_config.get('reshard_after_forward', True)

            if ep_enabled:
                _ensure_ep_fsdp_supported(model)

            # Collect experts map and expert params
            experts_map = _collect_ep_experts_map(model) if ep_enabled else {}
            expert_params = _collect_expert_params(model) if self.enable_ep else None

            # Build layer_pairs: [(layer_mod, experts_mod_or_None)]
            layers = _get_decoder_layers(model)
            layer_pairs = []
            if layers is not None:
                for layer_mod in layers:
                    experts_mod = _find_experts_in_layer(layer_mod, experts_map)
                    layer_pairs.append((layer_mod, experts_mod))

            # FSDP2 wrapping per layer
            world_size = self.device_mesh.world_size
            ep_fsdp_mesh_1d = self.ep_fsdp_device_mesh['ep_fsdp'] if ep_enabled else None

            for layer_mod, experts_mod in layer_pairs:
                layer_mod._fsdp_modules = []

                if experts_mod is not None and ep_fsdp_mesh_1d is not None:
                    from torch.distributed.tensor import Shard

                    # PreMulSum (used by set_gradient_divide_factor) only supports
                    # float16/float32/float64; override reduce_dtype to float32
                    # when the base policy uses bfloat16.
                    ep_mp_policy = _build_ep_mp_policy(mp_policy)
                    fully_shard(
                        experts_mod,
                        mesh=ep_fsdp_mesh_1d,
                        reshard_after_forward=reshard_after_forward,
                        mp_policy=ep_mp_policy,
                        shard_placement_fn=lambda param: Shard(1),
                    )
                    # gradient_divide_factor = world_size
                    experts_mod.set_gradient_divide_factor(world_size)
                    layer_mod._fsdp_modules.append(experts_mod)

                fully_shard(
                    layer_mod,
                    mesh=fsdp_mesh,
                    reshard_after_forward=reshard_after_forward,
                    mp_policy=mp_policy,
                    ignored_params=expert_params,
                )
                layer_mod._fsdp_modules.append(layer_mod)

            # Root model
            fully_shard(
                model,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
                ignored_params=expert_params,
            )

            # Manual prefetch
            if ep_enabled and layer_pairs:
                _setup_manual_prefetch([lp[0] for lp in layer_pairs])

            # Rebuild groups after wrapping so grad clip sees the live Parameter objects.
            if ep_enabled:
                _rebuild_ep_param_groups(model)

        if optimizer is not None:
            optimizer = _rebind_optimizer(optimizer, model)

        return model, optimizer

    def unwrap_model(self, model):
        return model


def _build_mp_policy(mixed_precision: str) -> 'MixedPrecisionPolicy':
    from torch.distributed.fsdp import MixedPrecisionPolicy
    if mixed_precision == 'bf16':
        dtype = torch.bfloat16
    elif mixed_precision == 'fp16':
        dtype = torch.float16
    else:
        return MixedPrecisionPolicy()
    return MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
        cast_forward_inputs=True,
    )


def _build_ep_mp_policy(base_policy: 'MixedPrecisionPolicy') -> 'MixedPrecisionPolicy':
    """Build a MixedPrecisionPolicy for EP experts with reduce_dtype=float32.

    NCCL's PreMulSum (used by set_gradient_divide_factor) only supports
    float16/float32/float64. When the base policy uses bfloat16 as reduce_dtype,
    we must override it to float32 for the expert FSDP group.
    """
    from torch.distributed.fsdp import MixedPrecisionPolicy
    reduce_dtype = base_policy.reduce_dtype
    if reduce_dtype == torch.bfloat16:
        reduce_dtype = torch.float32
    return MixedPrecisionPolicy(
        param_dtype=base_policy.param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=base_policy.output_dtype,
        cast_forward_inputs=base_policy.cast_forward_inputs,
    )


def _build_fsdp_mesh(device_mesh: DeviceMesh) -> Optional[TorchDeviceMesh]:
    if device_mesh is None or device_mesh.mesh_dim_names is None:
        return None
    flat_mesh = device_mesh.mesh.flatten()
    if flat_mesh.size <= 1:
        return None
    return TorchDeviceMesh(device_mesh.device_type, flat_mesh, mesh_dim_names=('fsdp', ))


def _get_decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    inner_model = getattr(model, 'model', None)
    if inner_model is not None:
        inner_layers = getattr(inner_model, 'layers', None)
        if isinstance(inner_layers, nn.ModuleList):
            return inner_layers

    return None


def _collect_expert_params(model: nn.Module) -> Optional[Set[nn.Parameter]]:
    ignored: Set[nn.Parameter] = set()
    ep_patched = False
    for module in model.modules():
        experts = getattr(module, 'experts', None)
        if experts is not None and getattr(module, '_ep_patched', False):
            ep_patched = True
            if isinstance(experts, nn.ModuleList):
                for expert in experts:
                    ignored.update(expert.parameters())
            else:
                ignored.update(experts.parameters())

        if getattr(module, '_ep_ignore_shared_experts', False) and getattr(module, '_ep_patched', False):
            ep_patched = True
            shared = getattr(module, 'shared_expert', None)
            if shared is not None:
                ignored.update(shared.parameters())

    if not ep_patched:
        return None
    return ignored or None


def _rebuild_ep_param_groups(model: nn.Module) -> None:
    expert_params = _collect_expert_params(model)
    if not expert_params:
        if hasattr(model, '_ep_param_groups'):
            delattr(model, '_ep_param_groups')
        return

    all_params = set(model.parameters())
    model._ep_param_groups = {
        'ep': list(expert_params),
        'non_ep': [p for p in all_params if p not in expert_params],
    }


def _collect_ep_experts_map(model: nn.Module) -> Dict[str, nn.Module]:
    """Collect {fqn: experts_module} for all EP-patched MoE blocks."""
    experts_map = {}
    for fqn, module in model.named_modules():
        if not getattr(module, '_ep_patched', False):
            continue
        experts = getattr(module, 'experts', None)
        if experts is not None:
            experts_fqn = fqn + '.experts' if fqn else 'experts'
            experts_map[experts_fqn] = experts
    return experts_map


def _find_experts_in_layer(layer_mod: nn.Module, experts_map: Dict[str, nn.Module]) -> Optional[nn.Module]:
    """Find the experts module inside a decoder layer, if any."""
    for module in layer_mod.modules():
        if module in experts_map.values():
            return module
    return None


def _setup_manual_prefetch(blocks: list) -> None:
    """Configure forward/backward prefetch for FSDP modules."""
    for i, block in enumerate(blocks):
        if i + 1 < len(blocks):
            next_fsdp_modules = getattr(blocks[i + 1], '_fsdp_modules', [])
            if next_fsdp_modules:
                block.set_modules_to_forward_prefetch(list(reversed(next_fsdp_modules)))
    for i in range(len(blocks) - 1, 0, -1):
        prev_fsdp_modules = getattr(blocks[i - 1], '_fsdp_modules', [])
        if prev_fsdp_modules:
            blocks[i].set_modules_to_backward_prefetch(list(reversed(prev_fsdp_modules)))


def _place_ep_experts_on_local_device(model: nn.Module, ep_fsdp_device_mesh: Optional[TorchDeviceMesh]) -> None:
    if ep_fsdp_device_mesh is None:
        return
    ep_world_size = ep_fsdp_device_mesh['ep'].size()
    if ep_world_size <= 1:
        return
    local_device = torch.device(Platform.get_local_device())
    for module in model.modules():
        if not getattr(module, '_ep_patched', False):
            continue
        experts = getattr(module, 'experts', None)
        if experts is not None:
            experts.to(local_device)
        if getattr(module, '_ep_ignore_shared_experts', False):
            shared = getattr(module, 'shared_expert', None)
            if shared is not None:
                shared.to(local_device)


def _ensure_moe_patched_if_needed(model: nn.Module, ep_fsdp_device_mesh: Optional[TorchDeviceMesh]) -> None:
    if ep_fsdp_device_mesh is None:
        return
    ep_world_size = ep_fsdp_device_mesh['ep'].size()
    if ep_world_size <= 1:
        return
    for module in model.modules():
        experts = getattr(module, 'experts', None)
        is_moe_experts = (
            isinstance(experts, nn.ModuleList) or (hasattr(experts, 'gate_up_proj') and hasattr(experts, 'down_proj')))
        if is_moe_experts and not getattr(module, '_ep_patched', False):
            raise RuntimeError('Found MoE experts but expert parallel is not applied. '
                               'Call apply_expert_parallel(model, device_mesh, config) before wrapping with FSDP2.')


def _ensure_ep_fsdp_supported(model: nn.Module) -> None:
    for module in model.modules():
        if not getattr(module, '_ep_patched', False):
            continue
        experts = getattr(module, 'experts', None)
        if isinstance(experts, nn.ModuleList):
            raise NotImplementedError('EP+EP_FSDP currently does not support MoE experts stored as nn.ModuleList. '
                                      'Only tensor experts (gate_up_proj/down_proj) are supported.')


def _rebind_optimizer(optimizer: torch.optim.Optimizer, model: nn.Module) -> torch.optim.Optimizer:
    if optimizer.state:
        raise RuntimeError('Optimizer already has state. Create the optimizer after FSDP wrapping, '
                           'or reinitialize it before training.')
    name_to_param = dict(model.named_parameters())
    ep_patched = any(getattr(module, '_ep_patched', False) for module in model.modules())
    if len(optimizer.param_groups) != 1:
        for group in optimizer.param_groups:
            if 'param_names' not in group:
                raise RuntimeError('NativeFSDPStrategy cannot rebind optimizer param_groups without param_names. '
                                   'Create the optimizer after wrapping, or include param_names in each group.')
            new_params = []
            for name in group['param_names']:
                if name not in name_to_param:
                    if ep_patched and '.experts.' in name:
                        continue
                    raise RuntimeError(
                        f"NativeFSDPStrategy could not find parameter '{name}' when rebinding optimizer.")
                new_params.append(name_to_param[name])
            group['params'] = new_params
        return optimizer
    optimizer.param_groups[0]['params'] = list(model.parameters())
    return optimizer
