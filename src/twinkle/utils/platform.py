from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Type, Optional
from typing import Union, List
import numpy


@dataclass
class DeviceMesh:
    device_type: str
    mesh: numpy.ndarray
    mesh_dim_names: Optional[tuple[str, ...]]

    def to_torch_device_mesh(self):
        import torch
        return torch.distributed.DeviceMesh(self.device_type, self.mesh, mesh_dim_names=self.mesh_dim_names)

    @property
    def dp_rank(self) -> int:
        return 0

    @property
    def sp_rank(self) -> int:
        return 0

    @property
    def cp_rank(self) -> int:
        return 0

    @property
    def mp_rank(self) -> int:
        return 0

    @property
    def pp_rank(self) -> int:
        return 0

    @property
    def ep_rank(self) -> int:
        return 0

    @property
    def dp_world_size(self) -> int:
        return 1

    @property
    def sp_world_size(self) -> int:
        return 1

    @property
    def cp_world_size(self) -> int:
        return 1

    @property
    def mp_world_size(self) -> int:
        return 1

    @property
    def pp_world_size(self) -> int:
        return 1

    @property
    def ep_world_size(self) -> int:
        return 1

    @property
    def world_size(self) -> int:
        return 1

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
    def get_platform(platform: str) -> Type['Platform']:
        if platform.upper() == "GPU":
            return GPU
        elif platform.upper() == "NPU":
            return NPU
        else:
            raise ValueError(f"Unsupported platform: {platform}")


class GPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'cuda'


class NPU(Platform):

    @staticmethod
    def visible_device_env():
        return 'ASCEND_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu:'
