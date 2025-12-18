from abc import ABC, abstractmethod
from typing import Literal
from functools import lru_cache


class Framework(ABC):

    @staticmethod
    @abstractmethod
    def get_library(module) -> str:
        """Get the library name of the input module"""
        ...

    @staticmethod
    @abstractmethod
    def get_current_device():
        """Set the current device"""
        ...

    @staticmethod
    @abstractmethod
    def get_device():
        """Get the device type"""
        ...

    @staticmethod
    @abstractmethod
    def set_device(idx: int):
        """Set the current device"""
        ...



class Torch(Framework):

    @staticmethod
    def get_library(module) -> Literal['transformers', 'megatron', 'other']:
        module_path = type(module).__module__
        if "transformers" in module_path:
            return "transformers"
        elif "megatron" in module_path:
            return "megatron"
        else:
            return "other"

    @lru_cache
    @staticmethod
    def is_torch_npu_available(check_device=False) -> bool:
        "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
        if not _torch_available or importlib.util.find_spec("torch_npu") is None:
            return False

        import torch
        import torch_npu  # noqa: F401

        if check_device:
            try:
                # Will raise a RuntimeError if no NPU is found
                _ = torch.npu.device_count()
                return torch.npu.is_available()
            except RuntimeError:
                return False
        return hasattr(torch, "npu") and torch.npu.is_available()

    @staticmethod
    @lru_cache
    def get_current_device():
        import torch
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        elif


    @staticmethod
    def get_device():
        pass

    @staticmethod
    def set_device(idx: int):
        pass


def get_framework(module) -> Literal['torch', 'other']:
    if "torch" in type(module).__module__ or hasattr(module, "parameters"):
        return "torch"
    return 'other'


def get_library(module) -> Literal['transformers', 'megatron', 'other']:
    """Get The library of one module

    Args:
        module: A torch.nn.Module instance

    Returns:
        A string representing the library, supports `transformers` or `megatron` or `other`
    """
    if get_framework(module) == 'torch':
        return Torch.get_library(module)
    return 'other'
