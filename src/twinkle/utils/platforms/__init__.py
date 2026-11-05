from .base import Platform
from .gpu import GPU
from .metax import Metax, is_metax_available
from .mps import MPS, is_mps_available
from .npu import NPU, ensure_hccl_socket_env, ensure_npu_backend, hccl_socket_env
from .xpu import XPU, ensure_bkcl_nonuniform_env, ensure_xpu_compat

# Run the detected platform's one-shot runtime fixups. This has to stay an
# import-time side effect: the vLLM EngineCore subprocess only *imports*
# twinkle.utils (through the worker extension) and never calls into Platform,
# yet it still needs XPU's ``torch.xpu`` stub to be installed before vLLM
# probes for an Intel XPU. No-op on GPU/NPU/Metax/MPS.
Platform.get_platform().bootstrap()
