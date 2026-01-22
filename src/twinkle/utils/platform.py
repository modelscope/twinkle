# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Dict
from typing import Type
from typing import Union, List
import torch.distributed as dist
import numpy as np


@dataclass
class DeviceMesh:
    """
    - dp: Data Parallel
    - fsdp: Fully Sharded Data Parallel
    - tp: Tensor Parallel
    - pp: Pipeline Parallel
    - ulysses: ulysses sequence parallel
    - cp: Context Parallel
    - ep: Expert Parallel
    - vpp: Virtual Pipeline Parallel

    Examples:
        # 8 GPUs: fsdp=4, tp=2
        mesh = DeviceMesh(
            mesh=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            mesh_dim_names=("fsdp", "tp"),
        )

        # 16 GPUs: dp=2, fsdp=2, tp=2, pp=2
        mesh = DeviceMesh(
            mesh=np.arange(16).reshape(2, 2, 2, 2),
            mesh_dim_names=("dp", "fsdp", "tp", "pp"),
        )
    """
    mesh: np.ndarray
    mesh_dim_names: Optional[tuple[str, ...]]
    ep_size: Optional[int] = None
    vpp_size: Optional[int] = None
    ulysses_size: Optional[int] = None
    device_type: str = 'cuda'

    @staticmethod
    def from_sizes(dp_size: int = 1, fsdp_size: int = None, tp_size: int = None,
                   pp_size: int = None, ulysses_size: int = None, cp_size: int = None, ep_size: int = None,
                   vpp_size: int = None, device_type: str = 'cuda') -> "DeviceMesh":

        origin_world_size = Platform.get_world_size()
        world_size = origin_world_size
        mesh_dim_names = []
        mesh_dim_sizes = []
        if fsdp_size is not None:
            mesh_dim_sizes.append(fsdp_size)
            mesh_dim_names.append("fsdp")
            if origin_world_size == 1:
                world_size *= fsdp_size
        if pp_size is not None:
            mesh_dim_sizes.append(pp_size)
            mesh_dim_names.append("pp")
            if origin_world_size == 1:
                world_size *= pp_size
        if dp_size is not None:
            mesh_dim_sizes.append(dp_size)
            mesh_dim_names.append("dp")
            if origin_world_size == 1:
                world_size *= dp_size
        if cp_size is not None:
            mesh_dim_sizes.append(cp_size)
            mesh_dim_names.append("cp")
            if origin_world_size == 1:
                world_size *= cp_size
        if tp_size is not None:
            mesh_dim_sizes.append(tp_size)
            mesh_dim_names.append("tp")
            if origin_world_size == 1:
                world_size *= tp_size

        """
        fsdp/dp/pp/tp/cp(megatron only) will affect the world_size
        sp here for two meanings:
        1. for transformers sp, only affect input data
        2. for megatron sp, limited by the tp
        vpp affect the macro-batch and inner loop
        ep affect the export groups
        """
        return DeviceMesh(
            device_type=device_type,
            mesh=np.arange(world_size).reshape(mesh_dim_sizes),
            mesh_dim_names=tuple(mesh_dim_names),
            vpp_size=vpp_size,
            ep_size=ep_size,
            ulysses_size=ulysses_size,
        )

    def __post_init__(self):
        if not isinstance(self.mesh, np.ndarray):
            self.mesh = np.array(self.mesh)

        valid_dim_names = {"dp", "fsdp", "tp", "pp", "sp", "cp"}
        if self.mesh_dim_names is not None:
            if len(self.mesh_dim_names) != len(self.mesh.shape):
                raise ValueError(
                    f"The shape of `mesh_dim_names`:({len(self.mesh_dim_names)}) "
                    f"does not match the shape of `mesh`: ({len(self.mesh.shape)})"
                )
        assert all([name in valid_dim_names for name in self.mesh_dim_names])

    def create_process_group(self, dims):
        rank = dist.get_rank()
        coords = np.argwhere(self.mesh == rank)[0]
        slices = []
        for i, dim_name in enumerate(self.mesh_dim_names):
            if dim_name in dims:
                slices.append(slice(None))
            else:
                slices.append(coords[i])

        ranks = sorted(self.mesh[tuple(slices)].flatten().tolist())
        return dist.new_group(ranks=ranks)

    def to_torch_device_mesh(self):
        import torch
        return torch.distributed.DeviceMesh(
            self.device_type,
            self.mesh,
            mesh_dim_names=self.mesh_dim_names
        )

    def _get_coord(self) -> Optional[tuple[int, ...]]:
        rank = Platform.get_rank()
        coords = np.argwhere(self.mesh == rank)
        if len(coords) == 0:
            return None
        return tuple(coords[0])

    def _get_coord_for_rank(self, rank: int) -> Optional[tuple[int, ...]]:
        coords = np.argwhere(self.mesh == rank)
        if len(coords) == 0:
            return None
        return tuple(coords[0])

    def _get_dim_index(self, dim_name: str) -> Optional[int]:
        if self.mesh_dim_names is None:
            return None
        if dim_name not in self.mesh_dim_names:
            return None
        return self.mesh_dim_names.index(dim_name)

    def _has_dim(self, dim_name: str) -> bool:
        return self._get_dim_index(dim_name) is not None

    def _get_rank_for_dim(self, dim_name: str) -> Optional[int]:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return None
        coord = self._get_coord()
        if coord is not None:
            return coord[dim_idx]
        else:
            return None

    def _get_world_size_for_dim(self, dim_name: str) -> Optional[int]:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return None
        return self.mesh.shape[dim_idx]

    @property
    def is_single_process(self) -> bool:
        return self.world_size == 1 and 'RANK' not in os.environ

    @property
    def dp_rank(self) -> Optional[int]:
        return self._get_rank_for_dim("dp")

    @property
    def fsdp_rank(self) -> Optional[int]:
        return self._get_rank_for_dim("fsdp")

    @property
    def tp_rank(self) -> Optional[int]:
        return self._get_rank_for_dim("tp")

    @property
    def pp_rank(self) -> Optional[int]:
        return self._get_rank_for_dim("pp")

    @property
    def cp_rank(self) -> Optional[int]:
        return self._get_rank_for_dim("cp")

    @property
    def dp_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("dp")

    @property
    def fsdp_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("fsdp")

    @property
    def tp_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("tp")

    @property
    def pp_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("pp")

    @property
    def cp_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("cp")

    @property
    def world_size(self) -> int:
        return self.mesh.flatten().shape[0]

    @property
    def data_rank(self) -> Optional[int]:
        dp_rank = self.dp_rank
        fsdp_rank = self.fsdp_rank
        fsdp_world_size = self.fsdp_world_size

        data_rank = dp_rank
        if fsdp_world_size is not None and fsdp_world_size > 1:
            if dp_rank is not None and fsdp_rank is not None:
                data_rank = dp_rank * fsdp_world_size + fsdp_rank
            elif fsdp_rank is not None:
                data_rank = fsdp_rank

        ulysses_size = self.ulysses_size or 1
        return data_rank // ulysses_size

    @property
    def data_world_size(self) -> int:
        dp_world_size = self.dp_world_size
        fsdp_world_size = self.fsdp_world_size
        if fsdp_world_size is not None and fsdp_world_size > 1:
            if dp_world_size is not None:
                return dp_world_size * fsdp_world_size
            else:
                return fsdp_world_size

        ulysses_size = self.ulysses_size or 1
        assert dp_world_size % ulysses_size == 0, f'dp_world_size: {dp_world_size} cannot be divided by ulysses_size: {ulysses_size}.'
        return dp_world_size // ulysses_size

    def get_slice(self, total_length: int, rank: Optional[int] = None) -> slice:
        world_size = self.data_world_size
        if rank is None:
            rank = self.data_rank
            if rank is None:
                rank = 0
                world_size = 1

        k, m = divmod(total_length, world_size)
        start = rank * k + min(rank, m)
        end = (rank + 1) * k + min(rank + 1, m)
        return slice(start, end)

    def get_slice_for_dim(self, dim_name: str, total_length: int) -> slice:
        world_size = self._get_world_size_for_dim(dim_name)
        rank = self._get_rank_for_dim(dim_name)
        if rank is None:
            raise ValueError(f'{dim_name} rank does not exist.')

        k, m = divmod(total_length, world_size)
        start = rank * k + min(rank, m)
        end = (rank + 1) * k + min(rank + 1, m)
        return slice(start, end)

    def get_pp_stage_ranks(self, stage: int) -> list[int]:
        pp_dim_idx = self._get_dim_index("pp")

        if pp_dim_idx is None:
            if stage == 0:
                return self.mesh.flatten().tolist()
            raise ValueError("No PP dimension, only stage 0 exists")

        indices = [slice(None)] * len(self.mesh.shape)
        indices[pp_dim_idx] = stage

        return sorted(self.mesh[tuple(indices)].flatten().tolist())

    def get_pp_first_ranks(self) -> list[int]:
        return self.get_pp_stage_ranks(0)

    def get_pp_last_ranks(self) -> list[int]:
        pp_world_size = self.pp_world_size or 1
        return self.get_pp_stage_ranks(pp_world_size - 1)

    def get_ranks_in_dim(self, dim_name: str) -> list[int]:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return [Platform.get_rank()]

        coord = list(self._get_coord())
        ranks = []
        for i in range(self.mesh.shape[dim_idx]):
            coord[dim_idx] = i
            ranks.append(int(self.mesh[tuple(coord)]))
        return ranks

    def get_submesh(self, dim_name: str) -> "DeviceMesh":
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            raise ValueError(f"Dimension '{dim_name}' not found in mesh_dim_names")

        coord = self._get_coord()
        indices = []
        for i, c in enumerate(coord):
            if i == dim_idx:
                indices.append(slice(None))
            else:
                indices.append(c)

        sub_mesh = self.mesh[tuple(indices)]
        return DeviceMesh(
            device_type=self.device_type,
            mesh=sub_mesh,
            mesh_dim_names=(dim_name,),
        )

    def __getitem__(self, dim_name: str) -> "DeviceMesh":
        return self.get_submesh(dim_name)

    def has_dim(self, dim_name: str) -> bool:
        if self.mesh_dim_names is None:
            return False
        return dim_name in self.mesh_dim_names

    def get_dim_size(self, dim_name: str) -> int:
        if not self.has_dim(dim_name):
            raise ValueError(f"Dimension '{dim_name}' not found in mesh. Available: {self.mesh_dim_names}")

        dim_idx = self.mesh_dim_names.index(dim_name)
        return self.mesh.shape[dim_idx]


@dataclass
class DeviceGroup:

    name: str
    ranks: Union[List[int], int]
    device_type: str
    _device_mesh: Dict[str, DeviceMesh] = field(default_factory=dict)


class Platform(ABC):

    @staticmethod
    def visible_device_env() -> str:
        return Platform.get_platform().visible_device_env()

    @staticmethod
    def device_prefix() -> str:
        return Platform.get_platform().device_prefix()

    @staticmethod
    def get_platform_names() -> List[str]:
        return ['GPU', 'NPU']

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
        if idx < 0:
            idx = 0
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
        # Ascend runtime uses ASCEND_RT_VISIBLE_DEVICES.
        return 'ASCEND_RT_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu'

    @staticmethod
    def get_local_device(idx: int, **kwargs) -> str:
        return f'npu:{idx}'
