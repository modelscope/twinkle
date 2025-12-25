import importlib
import os
from abc import ABC, abstractmethod
import random
from typing import Literal, Union, Optional
from functools import lru_cache

import numpy as np


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
    def get_rank() -> int:
        """Get the global rank"""
        return int(os.getenv('RANK', -1))

    @staticmethod
    def get_local_rank() -> int:
        """Get the local rank"""
        return int(os.getenv('LOCAL_RANK', -1))

    @staticmethod
    def get_world_size() -> int:
        """Get the world size"""
        return int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)

    @staticmethod
    def get_local_world_size() -> int:
        """Get the local world size"""
        return int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))

    @staticmethod
    def get_nnodes() -> int:
        """Get the node count"""
        return int(os.getenv('NNODES', 1))

    @staticmethod
    def get_node_rank() -> int:
        """Get the current node rank"""
        return int(os.getenv('NODE_RANK', 0))

    @staticmethod
    def is_local_master() -> bool:
        """Get if current is the local master"""
        local_rank = Framework.get_local_rank()
        return local_rank in {-1, 0}

    @staticmethod
    def is_master() -> bool:
        """Get if current is the global master"""
        rank = Framework.get_rank()
        return rank in {-1, 0}

    @staticmethod
    def is_last_rank() -> bool:
        """Get if current is the last rank"""
        rank = Framework.get_rank()
        world_size = Framework.get_world_size()
        return rank in {-1, world_size - 1}

    @staticmethod
    def get_framework(module) -> Literal['torch', 'other']:
        """Get the framework"""
        if "torch" in type(module).__module__ or hasattr(module, "parameters"):
            return "torch"
        return 'other'

    @staticmethod
    def get_library(module) -> Literal['transformers', 'megatron', 'other']:
        """Get The library of one module

        Args:
            module: A torch.nn.Module instance

        Returns:
            A string representing the library, supports `transformers` or `megatron` or `other`
        """
        if Framework.get_framework(module) == 'torch':
            return Torch.get_library(module)
        return 'other'

    @staticmethod
    def get_peer_index(target_size, rank=None, world_size=None):
        if rank is None:
            rank = Framework.get_rank()
        if rank < 0:
            rank = 0
        if world_size is None:
            world_size = Framework.get_world_size()
        if world_size <= 0:
            world_size = 1

        k, m = divmod(target_size, world_size)
        start_idx = rank * k + min(rank, m)
        end_idx = (rank + 1) * k + min(rank + 1, m)
        if target_size < world_size:
            start_idx = rank % target_size
            end_idx = start_idx + 1

        return slice(start_idx, end_idx)

    @staticmethod
    def seed_everything(seed: Optional[int] = 42, full_determinism: bool = False):
        Torch.seed_everything(seed, full_determinism)


class Torch(Framework):

    @staticmethod
    def get_library(module) -> Literal['transformers', 'megatron', 'other']:
        module_path = type(module).__module__
        if "transformers" in module_path:
            return "transformers"
        elif "megatron" in module_path:
            return "megatron"
        else:
            return "other"

    @staticmethod
    def is_torch_available() -> bool:
        """Check if `torch` is installed"""
        return importlib.util.find_spec('torch') is not None

    @staticmethod
    def is_torch_npu_available() -> bool:
        """Check if `torch_npu` is installed"""
        return importlib.util.find_spec('torch_npu') is not None

    @staticmethod
    def is_gpu_available() -> bool:
        "Checks if at least one GPU device is available"
        if not Torch.is_torch_available():
            return False

        import torch
        if not hasattr(torch, 'cuda'):
            return False

        return torch.cuda.is_available()

    @staticmethod
    def is_npu_available() -> bool:
        "Checks if `torch_npu` is installed and if at least one NPU device is available"
        if not Torch.is_torch_available() or not Torch.is_torch_npu_available():
            return False

        import torch
        import torch_npu
        if not hasattr(torch, 'npu'):
            return False

        return torch.npu.is_available() and torch.npu.device_count() > 0

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
    def get_device(local_rank) -> str:
        if local_rank is None:
            local_rank = max(0, Torch.get_local_rank())
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
    def set_device(local_rank: Union[int, str]) -> None:
        import torch
        if local_rank is None:
            local_rank = max(0, Torch.get_local_rank())
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
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
                os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                if Torch.is_npu_available():
                    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
                    os.environ["HCCL_DETERMINISTIC"] = "1"
