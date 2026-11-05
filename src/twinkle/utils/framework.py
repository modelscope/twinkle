# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import numpy as np
import random
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Union

from .device_mesh import DeviceMesh, Platform

if TYPE_CHECKING:
    import torch


class Framework(ABC):

    @staticmethod
    @abstractmethod
    def get_current_device() -> int:
        """Set the current device"""
        ...

    @staticmethod
    @abstractmethod
    def get_device(local_rank) -> str:
        """Get the device of the specified rank"""
        ...

    @staticmethod
    @abstractmethod
    def set_device(local_rank: Union[str, int]) -> None:
        """Set the current device"""
        ...

    @staticmethod
    def seed_everything(seed: Optional[int] = 42, full_determinism: bool = False):
        Torch.seed_everything(int(seed), full_determinism)

    @staticmethod
    def gather_object(object: Any, device_mesh: DeviceMesh, process_group=None):
        import torch.distributed as dist
        output_objects = [object]
        group_size = 1
        if dist.is_available() and dist.is_initialized():
            # Platforms whose device backend cannot carry object collectives (e.g.
            # HCCL) answer with a Gloo group; None keeps the caller's group.
            process_group = Platform.get_platform().resolve_object_collective_group(
                device_mesh) or process_group
            group_size = dist.get_world_size(group=process_group)
        if group_size > 1:
            output_objects = [None for _ in range(group_size)]
            dist.all_gather_object(output_objects, object, group=process_group)
        _x = []
        for y in output_objects:
            if y is None:
                continue
            if isinstance(y, (list, tuple)):
                _x.extend(y)
            else:
                _x.append(y)
        return _x


class Torch(Framework):

    @staticmethod
    @lru_cache
    def _device_module():
        """The current platform's torch.<x> device namespace, or ``None`` on CPU.

        Cached like the old availability probes: the platform is resolved once per
        process and device hot paths (``synchronize`` etc.) must not re-detect it.
        """
        return Platform.get_platform().device_module()

    @staticmethod
    @lru_cache
    def is_torch_available() -> bool:
        """Check if `torch` is installed"""
        return importlib.util.find_spec('torch') is not None

    @staticmethod
    @lru_cache
    def is_torch_npu_available() -> bool:
        """Check if `torch_npu` is installed"""
        return importlib.util.find_spec('torch_npu') is not None

    @staticmethod
    @lru_cache
    def is_gpu_available() -> bool:
        """Checks if at least one GPU device is available"""
        if not Torch.is_torch_available():
            return False

        import torch
        if not hasattr(torch, 'cuda'):
            return False

        return torch.cuda.is_available()

    @staticmethod
    @lru_cache
    def is_npu_available() -> bool:
        'Checks if `torch_npu` is installed and if at least one NPU device is available'
        if not Torch.is_torch_available() or not Torch.is_torch_npu_available():
            return False

        import torch
        import torch_npu
        if not hasattr(torch, 'npu'):
            return False

        return torch.npu.is_available() and torch.npu.device_count() > 0

    @staticmethod
    def empty_cache():
        device_module = Torch._device_module()
        if device_module is not None:
            device_module.empty_cache()

    @staticmethod
    @lru_cache
    def get_current_device() -> 'Union[int, str, "torch.device"]':
        device_module = Torch._device_module()
        if device_module is None:
            return 'cpu'
        return device_module.current_device()

    @staticmethod
    @lru_cache
    def get_device(local_rank) -> str:
        if local_rank is None:
            local_rank = max(0, Platform.get_local_rank())
        if Torch._device_module() is None:
            return 'cpu'
        return f'{Platform.device_prefix()}:{local_rank}'

    @staticmethod
    def set_device(local_rank: Union[int, str] = None) -> None:
        device_module = Torch._device_module()
        if device_module is None:
            return
        if local_rank is None:
            local_rank = max(0, Platform.get_local_rank())
        device_module.set_device(local_rank)

    @staticmethod
    def seed_everything(seed: Optional[int] = 42, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)

        device_module = Torch._device_module()
        if device_module is not None:
            device_module.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if device_module is not None:
                Platform.get_platform().apply_determinism()

    @staticmethod
    def to_local_tensor(tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Convert DTensor to local tensor if needed.

        Args:
            tensor: A torch.Tensor or DTensor instance.

        Returns:
            A local torch.Tensor.
        """
        if hasattr(tensor, 'full_tensor'):
            # DTensor from torch.distributed.tensor
            return tensor.full_tensor()
        elif hasattr(tensor, 'to_local'):
            # Alternative DTensor API
            return tensor.to_local()
        return tensor

    @staticmethod
    def synchronize(device=None):
        """Wait for all queued work on ``device``, defaulting to the LOCAL_RANK device.

        Pass ``device`` explicitly in subprocesses (e.g. vLLM TP workers) that inherit the
        parent's LOCAL_RANK but are bound to a different device.
        """
        device_module = Torch._device_module()
        if device_module is None:
            return
        if device is None:
            device = Platform.get_local_device()
        device_module.synchronize(device)

    @staticmethod
    def contains_nan(*args, **kwargs) -> bool:
        import torch

        def _check(obj: Any) -> bool:
            if isinstance(obj, torch.Tensor):
                return torch.isnan(obj).any().item()

            if isinstance(obj, dict):
                return any(_check(v) for v in obj.values())

            if isinstance(obj, (list, tuple, set)):
                return any(_check(item) for item in obj)

            return False

        for arg in args:
            if _check(arg):
                return True

        for value in kwargs.values():
            if _check(value):
                return True

        return False

    @staticmethod
    def is_accelerator_tensor(tensor) -> bool:
        """True for tensors living on an accelerator (cuda-like or npu)."""
        return tensor.is_cuda or getattr(tensor, 'is_npu', False)

    @staticmethod
    def ipc_collect():
        device_module = Torch._device_module()
        if device_module is not None:
            device_module.ipc_collect()
