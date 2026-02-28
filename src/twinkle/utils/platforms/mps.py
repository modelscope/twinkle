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
