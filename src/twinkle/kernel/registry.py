# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module registry."""
from typing import Dict, Any, Optional, Type, List
from logging import getLogger

from .base import DeviceType, is_kernels_available

logger = getLogger(__name__)


class LayerRegistry:
    """层级别 kernel 注册表管理

    职责：
    1. 存储 kernel 注册信息（按 kernel_name -> device -> mode 组织）
    2. 提供查询接口（get, has, list）
    3. 同步到 HF kernels 的全局映射
    """

    def __init__(self):
        self._registry: Dict[str, Dict[DeviceType, Dict[Any, Any]]] = {}
        self._synced = False

    def register(self, kernel_name: str, repo_spec: Any, device: DeviceType = "cuda", mode: Any = None) -> None:
        """注册 kernel"""
        if kernel_name not in self._registry:
            self._registry[kernel_name] = {}
        if device not in self._registry[kernel_name]:
            self._registry[kernel_name][device] = {}
        self._registry[kernel_name][device][mode] = repo_spec
        # 新注册后需要重新同步
        self._synced = False
        logger.debug(f"Registered layer kernel: {kernel_name} for device: {device}, mode: {mode}")

    def get(self, kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> Optional[Any]:
        """获取已注册的 kernel spec"""
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
        """检查 kernel 是否已注册"""
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
        """列出所有已注册的 kernel 名称"""
        return list(self._registry.keys())

    def sync_to_hf_kernels(self) -> None:
        """将注册表同步到 HF kernels"""
        if self._synced or not self._registry:
            return

        if not is_kernels_available():
            return

        from kernels import register_kernel_mapping as hf_register_kernel_mapping

        # 先清空，再批量注册
        hf_register_kernel_mapping({}, inherit_mapping=False)
        for kernel_name, device_dict in self._registry.items():
            hf_mapping = {kernel_name: device_dict}
            hf_register_kernel_mapping(hf_mapping, inherit_mapping=True)
            logger.debug(f"Synced {kernel_name} to HF kernels")

        self._synced = True

    def _clear(self) -> None:
        """清空注册表（仅用于测试）"""
        self._registry.clear()
        self._synced = False


_global_layer_registry = LayerRegistry()


class ExternalLayerRegistry:
    """外部层映射管理: class -> kernel_name"""

    def __init__(self):
        self._map: Dict[Type, str] = {}

    def register(self, layer_class: Type, kernel_name: str) -> None:
        self._map[layer_class] = kernel_name
        logger.debug(f"Registered external layer: {layer_class.__name__} -> {kernel_name}")

    def get(self, layer_class: Type) -> Optional[str]:
        return self._map.get(layer_class)

    def has(self, layer_class: Type) -> bool:
        return layer_class in self._map

    def list_mappings(self) -> list[tuple[Type, str]]:
        return list(self._map.items())

    def _clear(self) -> None:
        """清空注册表（仅用于测试）"""
        self._map.clear()


_global_external_layer_registry = ExternalLayerRegistry()


# ===== LayerRegistry 导出的函数 =====

def register_layer(kernel_name: str, repo_spec: Any, device: DeviceType = "cuda", mode: Any = None) -> None:
    _global_layer_registry.register(kernel_name, repo_spec, device, mode)


def get_layer_spec(kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> Optional[Any]:
    return _global_layer_registry.get(kernel_name, device, mode)


def list_kernel_names() -> List[str]:
    return _global_layer_registry.list_kernel_names()


def has_kernel(kernel_name: str, device: Optional[DeviceType] = None, mode: Any = None) -> bool:
    return _global_layer_registry.has(kernel_name, device, mode)


# ===== ExternalLayerRegistry 导出的函数 =====

def register_external_layer(layer_class: Type, kernel_name: str) -> None:
    """注册外部层映射并调用 replace_kernel_forward_from_hub 添加 kernel_layer_name 属性"""
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


# ===== 获取全局注册表的函数 =====

def get_global_layer_registry() -> LayerRegistry:
    return _global_layer_registry


def get_global_external_layer_registry() -> ExternalLayerRegistry:
    return _global_external_layer_registry
