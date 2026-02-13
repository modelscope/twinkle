# Copyright (c) ModelScope Contributors. All rights reserved.

from .lora import LoraParallelLinear, dispatch_megatron

__all__ = [
    'LoraParallelLinear',
    'dispatch_megatron',
]
