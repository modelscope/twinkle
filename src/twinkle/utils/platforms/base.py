# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from abc import ABC
from typing import List, Type


class Platform(ABC):

    @staticmethod
    def visible_device_env(platform: str = None) -> str:
        return Platform.get_platform(platform).visible_device_env()

    @staticmethod
    def device_prefix(platform: str = None) -> str:
        return Platform.get_platform(platform).device_prefix()

    @staticmethod
    def get_platform_names() -> List[str]:
        return ['GPU', 'NPU', 'MPS']

    @staticmethod
    def get_platform(platform: str = None) -> Type['Platform']:
        if platform is None:
            from .mps import is_mps_available
            if shutil.which('npu-smi'):
                from .npu import NPU, ensure_npu_backend
                ensure_npu_backend()
                return NPU
            elif shutil.which('nvidia-smi'):
                from .gpu import GPU
                return GPU
            elif is_mps_available():
                from .mps import MPS
                return MPS
            else:
                from .gpu import GPU
                return GPU
        elif platform.upper() in ('GPU', 'CUDA'):
            from .gpu import GPU
            return GPU
        elif platform.upper() == 'NPU':
            from .npu import NPU, ensure_npu_backend
            ensure_npu_backend()
            return NPU
        elif platform.upper() == 'MPS':
            from .mps import MPS
            return MPS
        else:
            raise ValueError(f'Unsupported platform: {platform}.')

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
        local_rank = Platform.get_local_rank()
        return local_rank in {-1, 0}

    @staticmethod
    def is_master() -> bool:
        """Get if current is the global master"""
        rank = Platform.get_rank()
        return rank in {-1, 0}

    @staticmethod
    def is_last_rank() -> bool:
        """Get if current is the last rank"""
        rank = Platform.get_rank()
        world_size = Platform.get_world_size()
        return rank in {-1, world_size - 1}

    @staticmethod
    def get_peer_index(target_size, rank=None, world_size=None):
        if rank is None:
            rank = Platform.get_rank()
        if rank < 0:
            rank = 0
        if world_size is None:
            world_size = Platform.get_world_size()
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
    def get_local_device(idx: int = None, *, platform: str = None):
        platform = Platform.get_platform(platform)
        if idx is None:
            idx = Platform.get_local_rank()
        if idx < 0:
            idx = 0
        return platform.get_local_device(idx)

    @staticmethod
    def device_backend(platform: str = None):
        platform = Platform.get_platform(platform)
        return platform.device_backend()

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0, platform=None) -> str:
        platform = Platform.get_platform(platform)
        return platform.get_vllm_device_uuid(device_id)
