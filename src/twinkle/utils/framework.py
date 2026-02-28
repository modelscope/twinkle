# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import numpy as np
import os
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
        import torch
        import torch.distributed as dist
        output_objects = [object]
        if device_mesh.data_world_size > 1:
            group_size = dist.get_world_size(group=process_group)
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
        if Torch.is_gpu_available():
            import torch
            torch.cuda.empty_cache()
        elif Torch.is_npu_available():
            import torch
            import torch_npu
            torch.npu.empty_cache()

    @staticmethod
    @lru_cache
    def get_current_device() -> 'Union[int, str, "torch.device"]':
        import torch
        if Torch.is_gpu_available():
            return torch.cuda.current_device()
        elif Torch.is_npu_available():
            import torch_npu
            return torch.npu.current_device()
        else:
            return 'cpu'

    @staticmethod
    @lru_cache
    def get_device(local_rank) -> str:
        if local_rank is None:
            local_rank = max(0, Platform.get_local_rank())
        local_rank = str(local_rank)
        if Torch.is_gpu_available():
            from .platform import GPU
            device = f'{GPU.device_prefix()}:{local_rank}'
        elif Torch.is_npu_available():
            from .platform import NPU
            device = f'{NPU.device_prefix()}:{local_rank}'
        else:
            device = 'cpu'
        return device

    @staticmethod
    def set_device(local_rank: Union[int, str] = None) -> None:
        import torch
        if local_rank is None:
            local_rank = max(0, Platform.get_local_rank())
        if Torch.is_gpu_available():
            torch.cuda.set_device(local_rank)
        elif Torch.is_npu_available():
            import torch_npu
            torch.npu.set_device(local_rank)

    @staticmethod
    def seed_everything(seed: Optional[int] = 42, deterministic: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        if Torch.is_gpu_available():
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if Torch.is_npu_available():
                import torch_npu
                torch.npu.manual_seed_all(seed)

            if deterministic:
                torch.use_deterministic_algorithms(True)
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
                os.environ['FLASH_ATTENTION_DETERMINISTIC'] = '1'
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                if Torch.is_npu_available():
                    os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
                    os.environ['HCCL_DETERMINISTIC'] = '1'

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
    def synchronize():
        import torch
        if Torch.is_gpu_available():
            torch.cuda.synchronize(Platform.get_local_device())
        elif Torch.is_npu_available():
            import torch_npu
            torch.npu.synchronize(Platform.get_local_device())

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
    def ipc_collect():
        if Torch.is_gpu_available():
            import torch
            torch.cuda.ipc_collect()
        elif Torch.is_npu_available():
            import torch
            import torch_npu
            torch.npu.ipc_collect()
