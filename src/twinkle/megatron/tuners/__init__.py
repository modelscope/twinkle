# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-compatible tuners for efficient fine-tuning."""

from .lora import LoraParallelLinear, dispatch_megatron

__all__ = [
    'LoraParallelLinear',
    'dispatch_megatron',
]
