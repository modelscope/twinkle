from .base import Platform
from .gpu import GPU
from .metax import Metax, is_metax_available
from .mps import MPS, is_mps_available
from .npu import NPU, ensure_hccl_socket_env, ensure_npu_backend, hccl_socket_env
