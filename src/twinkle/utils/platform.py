# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import platform
import shutil
import subprocess
from abc import ABC
from dataclasses import dataclass, field
from itertools import product
from functools import lru_cache
from typing import Optional, Dict
from typing import Type
from typing import Union, List
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
        # 8 GPUs: fsdp=4, dp=2
        mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)

        # 16 GPUs: dp=2, cp=2, tp=2, pp=2
        mesh = DeviceMesh.from_sizes(dp_size=2, cp_size=2, tp_size=2, pp_size=2)
    """
    mesh: np.ndarray
    mesh_dim_names: Optional[tuple[str, ...]]
    ep_size: Optional[int] = None
    etp_size: Optional[int] = None
    # megatron only
    vpp_size: Optional[int] = None
    # transformers only
    ulysses_size: Optional[int] = None
    # megatron only
    sequence_parallel: bool = False
    device_type: str = 'cuda'

    @staticmethod
    def from_sizes(*, world_size: int = 1, dp_size: int = 1, fsdp_size: int = None, tp_size: int = None,
                   pp_size: int = None, ulysses_size: int = None, cp_size: int = None, ep_size: int = None,
                   etp_size: int = None,vpp_size: int = None, device_type: str = 'cuda', sequence_parallel: bool = False) -> "DeviceMesh":
        """Create a default device mesh from the given sizes.

        Args:
            world_size: The global world size, can be referenced from other sizes
            dp_size: The data parallel size
            fsdp_size: The fsdp2 parallel size
            tp_size: The tensor parallel size
            pp_size: The pipeline parallel size
            ulysses_size: The ulysses parallel size
            cp_size: The context parallel size
            ep_size: The expert parallel size
            etp_size: The expert tensor parallel size
            vpp_size: The virtual pipeline parallel size
            device_type: The device type
            sequence_parallel: Use sequence parallel or not, default false
        Returns:
            The device mesh instance
        """

        origin_world_size = world_size
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
            mesh_dim_names.append("dp")
            if origin_world_size == 1:
                world_size *= dp_size
            mesh_dim_sizes.append(dp_size)
        else:
            mesh_dim_sizes.append(-1)
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
        return DeviceMesh(
            device_type=device_type,
            mesh=np.arange(world_size).reshape(mesh_dim_sizes),
            mesh_dim_names=tuple(mesh_dim_names),
            vpp_size=vpp_size,
            ep_size=ep_size,
            etp_size=etp_size,
            ulysses_size=ulysses_size,
            sequence_parallel=sequence_parallel,
        )

    def __post_init__(self):
        if not isinstance(self.mesh, np.ndarray):
            self.mesh = np.array(self.mesh)

        valid_dim_names = {"dp", "fsdp", "tp", "pp", "cp"}
        if self.mesh_dim_names is not None:
            if len(self.mesh_dim_names) != len(self.mesh.shape):
                raise ValueError(
                    f"The shape of `mesh_dim_names`:({len(self.mesh_dim_names)}) "
                    f"does not match the shape of `mesh`: ({len(self.mesh.shape)})"
                )
        assert all([name in valid_dim_names for name in self.mesh_dim_names])

    def create_process_group(self, dims):
        """Create a process group by dims"""
        import torch.distributed as dist
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

    def get_dim_group(self, dims):
        import torch.distributed as dist
        if isinstance(dims, str):
            dims = (dims,)
        if len(dims) != 1:
            return self.create_process_group(dims)

        dim_name = dims[0]
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            raise ValueError(f"Dimension '{dim_name}' not found in mesh_dim_names")

        cache = getattr(self, "_dim_group_cache", {})
        if dim_name in cache:
            coord = self._get_coord()
            key = tuple(c for i, c in enumerate(coord) if i != dim_idx)
            return cache[dim_name][key]

        other_shape = [self.mesh.shape[i] for i in range(self.mesh.ndim) if i != dim_idx]
        group_map = {}
        for other_coord in product(*[range(s) for s in other_shape]):
            ranks = []
            for dim_val in range(self.mesh.shape[dim_idx]):
                full_coord = []
                other_iter = iter(other_coord)
                for i in range(self.mesh.ndim):
                    if i == dim_idx:
                        full_coord.append(dim_val)
                    else:
                        full_coord.append(next(other_iter))
                ranks.append(int(self.mesh[tuple(full_coord)]))
            group = dist.new_group(ranks=ranks)
            group_map[other_coord] = group

        cache[dim_name] = group_map
        setattr(self, "_dim_group_cache", cache)

        coord = self._get_coord()
        key = tuple(c for i, c in enumerate(coord) if i != dim_idx)
        return group_map[key]

    @property
    def order(self):
        """The order of the dimensions for megatron"""
        # TODO hard coded for now
        return 'tp-cp-ep-dp-pp'

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

    def _get_world_size_for_dim(self, dim_name: str) -> int:
        dim_idx = self._get_dim_index(dim_name)
        if dim_idx is None:
            return 0 # not valid
        return self.mesh.shape[dim_idx]

    @property
    def is_single_process(self) -> bool:
        return self.world_size == 1 and 'RANK' not in os.environ

    @property
    def dp_rank(self) -> Optional[int]:
        rank = self._get_rank_for_dim("dp")
        return rank

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
    def ep_rank(self) -> Optional[int]:
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
    def cp_world_size(self) -> int:
        return self._get_world_size_for_dim("cp")

    @property
    def ep_world_size(self) -> Optional[int]:
        return self._get_world_size_for_dim("ep")

    @property
    def etp_world_size(self) -> int:
        if self.etp_size is not None:
            return self.etp_size
        return self.tp_world_size or 1

    @property
    def world_size(self) -> int:
        return self.mesh.flatten().shape[0]

    @property
    def data_rank(self) -> Optional[int]:
        """Consider all dp/fsdp ranks, uses to determine how to distribute the data"""
        dp_rank = self.dp_rank
        fsdp_rank = self.fsdp_rank
        fsdp_world_size = self.fsdp_world_size

        data_rank = dp_rank
        if fsdp_world_size is not None and fsdp_world_size > 1:
            if dp_rank is not None and fsdp_rank is not None:
                data_rank = dp_rank * fsdp_world_size + fsdp_rank
            elif fsdp_rank is not None:
                data_rank = fsdp_rank

        # megatron dp_size=1
        if data_rank is None:
            data_rank = 0

        ulysses_size = self.ulysses_size or 1
        if data_rank is None:
            return None
        return data_rank // ulysses_size
    
    def get_data_rank_from_global_rank(self, global_rank: int) -> int:
        """Consider all dp/fsdp ranks and get the data rank of the global_rank,
        uses to determine how to distribute the data in driver"""
        coord = self._get_coord_for_rank(global_rank)
        if coord is None:
            return 0
        
        dp_idx = self._get_dim_index("dp")
        fsdp_idx = self._get_dim_index("fsdp")
        
        dp_rank = coord[dp_idx] if dp_idx is not None else None
        fsdp_rank = coord[fsdp_idx] if fsdp_idx is not None else None
        fsdp_world_size = self.fsdp_world_size if fsdp_idx is not None else 0
        
        data_rank = dp_rank
        if fsdp_world_size > 1:
            if dp_rank is not None and fsdp_rank is not None:
                data_rank = dp_rank * fsdp_world_size + fsdp_rank
            elif fsdp_rank is not None:
                data_rank = fsdp_rank
        
        if data_rank is None:
            data_rank = 0
        
        ulysses_size = self.ulysses_size or 1
        return data_rank // ulysses_size

    @property
    def data_world_size(self) -> int:
        """Consider all dp/fsdp ranks, uses to determine how to distribute the data"""
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
        if world_size == 1:
            return slice(0, total_length)
        if rank is None:
            rank = self.data_rank
            if rank is None:
                rank = 0
                world_size = 1

        k, m = divmod(total_length, world_size)
        start = rank * k + min(rank, m)
        end = (rank + 1) * k + min(rank + 1, m)
        return slice(start, end)

    def is_pp_first_rank(self) -> bool:
        pp_ranks = self.get_pp_first_ranks()
        if pp_ranks is None:
            return False
        return Platform.get_rank() in pp_ranks

    def is_pp_last_rank(self) -> bool:
        pp_ranks = self.get_pp_last_ranks()
        if pp_ranks is None:
            return False
        return Platform.get_rank() in pp_ranks

    def get_pp_stage_ranks(self, stage: int) -> Optional[list[int]]:
        pp_dim_idx = self._get_dim_index("pp")

        if pp_dim_idx is None:
            if stage == 0:
                return self.mesh.flatten().tolist()
            raise None

        indices = [slice(None)] * len(self.mesh.shape)
        indices[pp_dim_idx] = stage

        return sorted(self.mesh[tuple(indices)].flatten().tolist())

    def get_pp_first_ranks(self) -> Optional[list[int]]:
        return self.get_pp_stage_ranks(0)

    def get_pp_last_ranks(self) -> Optional[list[int]]:
        pp_world_size = self.pp_world_size or 1
        return self.get_pp_stage_ranks(pp_world_size - 1)

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
    """The device group to create/use resources

    name: The name of the device group, should be unique.
    ranks: The ranks of the device group, for example, 16, list(range(16))
    device_type: The device_type of the device group
    gpus_per_worker: The number of GPUs allocated for one process
    _device_mesh: Do not use, only for show logs.
    """

    name: str
    ranks: Union[List[int], int]
    device_type: str
    gpus_per_worker: int = 1
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
        return ['GPU', 'NPU', 'MPS']

    @staticmethod
    def get_platform(platform: str = None) -> Type['Platform']:
        if platform is None:
            if shutil.which("npu-smi"):
                return NPU
            elif shutil.which("nvidia-smi"):
                return GPU
            elif MPS.is_mps_available():
                return MPS
            else:
                return GPU
        elif platform.upper() in ("GPU", "CUDA"):
            return GPU
        elif platform.upper() == "NPU":
            return NPU
        elif platform.upper() == "MPS":
            return MPS
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

    @staticmethod
    def device_backend(platform: str = None):
        platform = Platform.get_platform(platform)
        return platform.device_backend()

class GPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'cuda'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'cuda:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        return 'nccl'


class NPU(Platform):

    @staticmethod
    def visible_device_env():
        # Ascend runtime uses ASCEND_RT_VISIBLE_DEVICES.
        return 'ASCEND_RT_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'npu:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        return 'hccl'


class MPS(Platform):

    @staticmethod
    def visible_device_env():
        return None

    @staticmethod
    def device_prefix():
        return 'mps'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'mps'

    @staticmethod
    def device_backend(platform: str = None):
        return 'gloo'

    @lru_cache
    @staticmethod
    def is_mps_available():
        if platform.system() != "Darwin":
            return False
        try:
            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL, text=True
            )
            return "Metal Support" in output
        except Exception: # noqa
            return False

def is_last_rank():
    import torch.distributed as dist
    if not dist.is_initialized():
        return True
    return dist.get_rank() == dist.get_world_size() - 1

def is_master():
    return Platform.is_master()
