import os
import shutil
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Optional, Dict
from typing import Type
from typing import Union, List

import numpy as np


@dataclass
class DeviceMesh:
    device_type: str
    mesh: np.ndarray
    mesh_dim_names: Optional[tuple[str, ...]]

    def to_torch_device_mesh(self):
        import torch
        return torch.distributed.DeviceMesh(self.device_type, self.mesh, mesh_dim_names=self.mesh_dim_names)

    def _get_coord(self) -> tuple[int, ...]:
        coords = np.argwhere(self.mesh == Platform.get_rank())
        if len(coords) == 0:
            raise ValueError(f"Rank {Platform.get_rank()} not found in mesh")
        return tuple(coords[0])

    def _get_dim_index(self, dim_name: str) -> Optional[int]:
        if self.mesh_dim_names is None:
            return None
        if dim_name not in self.mesh_dim_names:
            return None
        return self.mesh_dim_names.index(dim_name)

    def _get_rank_for_dim(self, dim_name: str) -> int:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return 0
        coord = self._get_coord()
        return coord[dim_idx]

    def _get_world_size_for_dim(self, dim_name: str) -> int:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return 1
        return self.mesh.shape[dim_idx]

    @property
    def dp_rank(self) -> int:
        return self._get_rank_for_dim("dp")

    @property
    def sp_rank(self) -> int:
        return self._get_rank_for_dim("sp")

    @property
    def cp_rank(self) -> int:
        return self._get_rank_for_dim("cp")

    @property
    def tp_rank(self) -> int:
        return self._get_rank_for_dim("tp")

    @property
    def pp_rank(self) -> int:
        return self._get_rank_for_dim("pp")

    @property
    def ep_rank(self) -> int:
        return self._get_rank_for_dim("ep")

    @property
    def mp_rank(self) -> int:
        return self._get_rank_for_dim("tp")

    @property
    def dp_world_size(self) -> int:
        return self._get_world_size_for_dim("dp")

    @property
    def sp_world_size(self) -> int:
        return self._get_world_size_for_dim("sp")

    @property
    def cp_world_size(self) -> int:
        return self._get_world_size_for_dim("cp")

    @property
    def tp_world_size(self) -> int:
        return self._get_world_size_for_dim("tp")

    @property
    def pp_world_size(self) -> int:
        return self._get_world_size_for_dim("pp")

    @property
    def ep_world_size(self) -> int:
        return self._get_world_size_for_dim("ep")

    @property
    def mp_world_size(self) -> int:
        return self._get_world_size_for_dim("tp")

    @property
    def world_size(self) -> int:
        return Platform.get_world_size()

    def get_slice(self, total_length):
        length = self.dp_world_size
        dp_rank = self.dp_rank
        k, m = divmod(total_length, length)
        return slice(dp_rank * k + min(dp_rank, m), (dp_rank + 1) * k + min(dp_rank + 1, m))


@dataclass
class DeviceGroup:

    name: str
    ranks: Union[List[int], int]
    device_type: str
    _device_mesh: Dict[str, DeviceMesh] = field(default_factory=dict)


class Platform(ABC):

    @staticmethod
    @abstractmethod
    def visible_device_env() -> str:
        ...

    @staticmethod
    @abstractmethod
    def device_prefix() -> str:
        ...

    @staticmethod
    def get_platform(platform: str = None) -> Type['Platform']:
        if platform is None:
            if shutil.which("npu-smi"):
                return NPU
            elif shutil.which("nvidia-smi"):
                return GPU
            else:
                return GPU
        elif platform.upper() in ("GPU", "CUDA"):
            return GPU
        elif platform.upper() == "NPU":
            return NPU
        else:
            raise ValueError(f"Unsupported platform: {platform}.")

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
        return platform.get_local_device(idx)

class GPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'cuda'

    @staticmethod
    def get_local_device(idx: int, **kwargs) -> str:
        return f'cuda:{idx}'


class NPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'ASCEND_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu'

    @staticmethod
    def get_local_device(idx: int, **kwargs) -> str:
        return f'npu:{idx}'
