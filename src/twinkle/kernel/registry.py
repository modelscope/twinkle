# Copyright (c) ModelScope Contributors. All rights reserved.

from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Any, Optional, Type, List, Callable, Tuple

from .base import DeviceType, ModeType, is_kernels_available
from kernels.layer.func import FuncRepositoryProtocol



logger = getLogger(__name__)


class LayerRegistry:
    """Manages kernel registrations and syncs to HF kernels."""

    def __init__(self):
        self._registry: Dict[str, Dict[DeviceType, Dict[Any, Any]]] = {}
        self._synced = False

    def register(self, kernel_name: str, repo_spec: Any, device: DeviceType = "cuda", mode: Any = None) -> None:
        if kernel_name not in self._registry:
            self._registry[kernel_name] = {}
        if device not in self._registry[kernel_name]:
            self._registry[kernel_name][device] = {}
        self._registry[kernel_name][device][mode] = repo_spec
        self._synced = False

    def get(self, kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> Optional[Any]:
        if kernel_name not in self._registry:
            return None
        devices = self._registry[kernel_name]
        if device is None:
            device = next(iter(devices.keys()), None)
            if device is None:
                return None
        modes = devices.get(device)
        if modes is None:
            return None
        if mode is None:
            return next(iter(modes.values()), None)
        return modes.get(mode)

    def has(self, kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> bool:
        if kernel_name not in self._registry:
            return False
        devices = self._registry[kernel_name]
        if device is None:
            return True
        if device not in devices:
            return False
        if mode is None:
            return True
        return mode in devices[device]

    def list_kernel_names(self) -> List[str]:
        return list(self._registry.keys())

    def sync_to_hf_kernels(self) -> None:
        if self._synced or not self._registry:
            return

        if not is_kernels_available():
            return

        from kernels import register_kernel_mapping as hf_register_kernel_mapping

        hf_register_kernel_mapping({}, inherit_mapping=False)
        for kernel_name, device_dict in self._registry.items():
            hf_mapping = {kernel_name: device_dict}
            hf_register_kernel_mapping(hf_mapping, inherit_mapping=True)

        self._synced = True

    def _clear(self) -> None:
        self._registry.clear()
        self._synced = False


_global_layer_registry = LayerRegistry()


class ExternalLayerRegistry:
    """Maps layer classes to kernel names."""

    def __init__(self):
        self._map: Dict[Type, str] = {}

    def register(self, layer_class: Type, kernel_name: str) -> None:
        self._map[layer_class] = kernel_name

    def get(self, layer_class: Type) -> Optional[str]:
        return self._map.get(layer_class)

    def has(self, layer_class: Type) -> bool:
        return layer_class in self._map

    def list_mappings(self) -> List[Tuple[Type, str]]:
        return list(self._map.items())

    def _clear(self) -> None:
        self._map.clear()


_global_external_layer_registry = ExternalLayerRegistry()


@dataclass(frozen=True)
class FunctionKernelSpec:
    func_name: str
    target_module: str
    func_impl: Optional[Callable]
    repo: Optional[FuncRepositoryProtocol]
    repo_id: Optional[str]
    revision: Optional[str]
    version: Optional[str]
    device: Optional[str]
    mode: Optional[ModeType]



class FunctionRegistry:
    """Manages function-level kernel registrations."""

    def __init__(self) -> None:
        self._registry: List[FunctionKernelSpec] = []

    def register(self, spec: FunctionKernelSpec) -> None:
        if spec in self._registry:
            return
        self._registry.append(spec)

    def list_specs(self) -> List[FunctionKernelSpec]:
        return list(self._registry)

    def _clear(self) -> None:
        self._registry.clear()


_global_function_registry = FunctionRegistry()


def register_layer(kernel_name: str, repo_spec: Any, device: DeviceType = "cuda", mode: Any = None) -> None:
    _global_layer_registry.register(kernel_name, repo_spec, device, mode)


def get_layer_spec(kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> Optional[Any]:
    return _global_layer_registry.get(kernel_name, device, mode)


def list_kernel_names() -> List[str]:
    return _global_layer_registry.list_kernel_names()


def has_kernel(kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> bool:
    return _global_layer_registry.has(kernel_name, device, mode)


def register_external_layer(layer_class: Type, kernel_name: str) -> None:
    _global_external_layer_registry.register(layer_class, kernel_name)

    if is_kernels_available():
        from kernels import replace_kernel_forward_from_hub
        replace_kernel_forward_from_hub(layer_class, kernel_name)
        logger.info(f"Registered {layer_class.__name__} -> kernel: {kernel_name}")
    else:
        logger.warning(
            f"HF kernels not available. {layer_class.__name__} mapping registered "
            f"but kernel replacement will not work without kernels package."
        )


def get_external_kernel_name(layer_class: Type) -> Optional[str]:
    return _global_external_layer_registry.get(layer_class)


def get_global_layer_registry() -> LayerRegistry:
    return _global_layer_registry


def get_global_external_layer_registry() -> ExternalLayerRegistry:
    return _global_external_layer_registry


def get_global_function_registry() -> FunctionRegistry:
    return _global_function_registry
