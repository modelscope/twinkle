# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
from abc import abstractmethod, ABC
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
    - sp: Sequence Parallel
    - cp: Context Parallel
    - ep: Expert Parallel

    Examples:
        # 8 GPUs: fsdp=4, tp=2
        mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
            mesh_dim_names=("fsdp", "tp"),
        )

        # 16 GPUs: dp=2, fsdp=2, tp=2, pp=2
        mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(16).reshape(2, 2, 2, 2),
            mesh_dim_names=("dp", "fsdp", "tp", "pp"),
        )
    """
    device_type: str
    mesh: np.ndarray
    mesh_dim_names: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        if not isinstance(self.mesh, np.ndarray):
            self.mesh = np.array(self.mesh)

        valid_dim_names = {"dp", "fsdp", "tp", "pp", "sp", "cp", "ep"}
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

    def _get_coord(self) -> tuple[int, ...]:
        rank = Platform.get_rank()
        coords = np.argwhere(self.mesh == rank)
        if len(coords) == 0:
            raise ValueError(f"Rank {rank} not found in mesh")
        return tuple(coords[0])

    def _get_coord_for_rank(self, rank: int) -> tuple[int, ...]:
        coords = np.argwhere(self.mesh == rank)
        if len(coords) == 0:
            raise ValueError(f"Rank {rank} not found in mesh")
        return tuple(coords[0])

    def _get_dim_index(self, dim_name: str) -> Optional[int]:
        if self.mesh_dim_names is None:
            return None
        if dim_name not in self.mesh_dim_names:
            return None
        return self.mesh_dim_names.index(dim_name)

    def _has_dim(self, dim_name: str) -> bool:
        return self._get_dim_index(dim_name) is not None

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
    def is_single_process(self) -> bool:
        return self.world_size == 1 and 'RANK' not in os.environ

    @property
    def dp_rank(self) -> int:
        return self._get_rank_for_dim("dp")

    @property
    def fsdp_rank(self) -> int:
        return self._get_rank_for_dim("fsdp")

    @property
    def tp_rank(self) -> int:
        return self._get_rank_for_dim("tp")

    @property
    def pp_rank(self) -> int:
        return self._get_rank_for_dim("pp")

    @property
    def sp_rank(self) -> int:
        return self._get_rank_for_dim("sp")

    @property
    def cp_rank(self) -> int:
        return self._get_rank_for_dim("cp")

    @property
    def ep_rank(self) -> int:
        return self._get_rank_for_dim("ep")

    @property
    def dp_world_size(self) -> int:
        return self._get_world_size_for_dim("dp")

    @property
    def fsdp_world_size(self) -> int:
        return self._get_world_size_for_dim("fsdp")

    @property
    def tp_world_size(self) -> int:
        return self._get_world_size_for_dim("tp")

    @property
    def pp_world_size(self) -> int:
        return self._get_world_size_for_dim("pp")

    @property
    def sp_world_size(self) -> int:
        return self._get_world_size_for_dim("sp")

    @property
    def cp_world_size(self) -> int:
        return self._get_world_size_for_dim("cp")

    @property
    def ep_world_size(self) -> int:
        return self._get_world_size_for_dim("ep")

    @property
    def world_size(self) -> int:
        return self.mesh.flatten().shape[0]

    @property
    def mp_rank(self) -> int:
        return self.tp_rank

    @property
    def mp_world_size(self) -> int:
        return self.tp_world_size

    @property
    def data_parallel_rank(self) -> int:
        dp_rank = self.dp_rank
        fsdp_rank = self.fsdp_rank
        fsdp_world_size = self.fsdp_world_size

        return dp_rank * fsdp_world_size + fsdp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self.dp_world_size * self.fsdp_world_size

    def get_slice(self, total_length: int, rank: Optional[int] = None) -> slice:
        world_size = self.data_parallel_world_size
        if rank is None:
            rank = self.data_parallel_rank

        k, m = divmod(total_length, world_size)
        start = rank * k + min(rank, m)
        end = (rank + 1) * k + min(rank + 1, m)
        return slice(start, end)

    def get_slice_for_dim(self, dim_name: str, total_length: int) -> slice:
        world_size = self._get_world_size_for_dim(dim_name)
        rank = self._get_rank_for_dim(dim_name)

        k, m = divmod(total_length, world_size)
        start = rank * k + min(rank, m)
        end = (rank + 1) * k + min(rank + 1, m)
        return slice(start, end)

    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == self.pp_world_size - 1

    def get_pp_prev_rank(self) -> Optional[int]:
        if self.is_first_pp_stage:
            return None

        pp_idx = self._get_dim_index("pp")
        if pp_idx is None:
            return None

        coord = list(self._get_coord())
        coord[pp_idx] -= 1
        return int(self.mesh[tuple(coord)])

    def get_pp_next_rank(self) -> Optional[int]:
        if self.is_last_pp_stage:
            return None

        pp_idx = self._get_dim_index("pp")
        if pp_idx is None:
            return None

        coord = list(self._get_coord())
        coord[pp_idx] += 1
        return int(self.mesh[tuple(coord)])

    def get_pp_group_ranks(self) -> list[int]:
        pp_idx = self._get_dim_index("pp")
        if pp_idx is None:
            return [Platform.get_rank()]

        coord = list(self._get_coord())
        ranks = []
        for i in range(self.pp_world_size):
            coord[pp_idx] = i
            ranks.append(int(self.mesh[tuple(coord)]))
        return ranks

    def get_pp_layer_range(self, total_layers: int) -> tuple[int, int]:
        pp_rank = self.pp_rank
        pp_world_size = self.pp_world_size

        layers_per_stage = total_layers // pp_world_size
        remainder = total_layers % pp_world_size

        if pp_rank < remainder:
            start = pp_rank * (layers_per_stage + 1)
            end = start + layers_per_stage + 1
        else:
            start = remainder * (layers_per_stage + 1) + (pp_rank - remainder) * layers_per_stage
            end = start + layers_per_stage

        return start, end

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

    def get_fsdp_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("fsdp")

    def get_tp_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("tp")

    def get_dp_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("dp")

    def get_sp_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("sp")

    def get_cp_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("cp")

    def get_ep_group_ranks(self) -> list[int]:
        return self.get_ranks_in_dim("ep")

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

    def get_process_group(self, dim_name: str):
        torch_mesh = self.to_torch_device_mesh()
        return torch_mesh.get_group(dim_name)

    def shares_data_with(self, other_rank: int) -> bool:
        other_coord = self._get_coord_for_rank(other_rank)

        dp_idx = self._get_dim_index("dp")
        fsdp_idx = self._get_dim_index("fsdp")

        other_dp_rank = other_coord[dp_idx] if dp_idx is not None else 0
        other_fsdp_rank = other_coord[fsdp_idx] if fsdp_idx is not None else 0
        other_fsdp_world_size = self.fsdp_world_size

        other_data_parallel_rank = other_dp_rank * other_fsdp_world_size + other_fsdp_rank

        return self.data_parallel_rank == other_data_parallel_rank

    def has_dim(self, dim_name: str) -> bool:
        if self.mesh_dim_names is None:
            return False
        return dim_name in self.mesh_dim_names

    def get_dim_size(self, dim_name: str) -> int:
        if not self.has_dim(dim_name):
            raise ValueError(f"Dimension '{dim_name}' not found in mesh. Available: {self.mesh_dim_names}")

        dim_idx = self.mesh_dim_names.index(dim_name)
        return self.mesh.shape[dim_idx]

    def get_dim_group(self, dim_name: str):
        if not self.has_dim(dim_name):
            return None
        torch_mesh = self.to_torch_device_mesh()
        return torch_mesh.get_group(dim_name)

    def get_local_rank_in_dim(self, dim_name: str) -> int:
        return self._get_rank_for_dim(dim_name)


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
        return int(os.getenv('RANK', 0))

    @staticmethod
    def get_local_rank() -> int:
        """Get the local rank"""
        return int(os.getenv('LOCAL_RANK', 0))

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
