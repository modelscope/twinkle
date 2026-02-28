# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Platform


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

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        from vllm.platforms import current_platform
        return current_platform.get_device_uuid(device_id)
