from .base import Platform
from .gpu import GPU
from .mps import MPS, is_mps_available
from .npu import NPU, ensure_hccl_socket_env, ensure_npu_backend
