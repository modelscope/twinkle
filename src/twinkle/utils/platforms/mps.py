# Copyright (c) ModelScope Contributors. All rights reserved.
import platform
import subprocess
from functools import lru_cache

from .base import Platform


@lru_cache
def is_mps_available():
    if platform.system() != 'Darwin':
        return False
    try:
        output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'],
                                         stderr=subprocess.DEVNULL,
                                         text=True)
        return 'Metal Support' in output
    except Exception:  # noqa
        return False


class MPS(Platform):

    @staticmethod
    def visible_device_env():
        return None

    @staticmethod
    def device_prefix():
        return 'mps'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return 'mps'

    @staticmethod
    def device_backend(platform: str = None):
        return 'gloo'

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        raise NotImplementedError

    @staticmethod
    def get_device_rng_state():
        import torch
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'get_rng_state'):
            try:
                return torch.mps.get_rng_state()
            except Exception:  # noqa: BLE001
                return None
        return None

    @staticmethod
    def set_device_rng_state(state) -> None:
        import torch
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'set_rng_state'):
            try:
                torch.mps.set_rng_state(state)
            except Exception:  # noqa: BLE001
                pass
