# Copyright (c) twinkle authors. All rights reserved.
"""Megatron model initialization and weight conversion.

This module provides independent implementation for weight loading/saving,
without external dependencies on swift.
"""

from .bridge import (
    # Main classes
    TwinkleBridgeAdapter,
    TwinkleGPTBridge,
    BridgeConfig,
    SafetensorLoader,
    StreamingSafetensorSaver,
    LazyTensor,
    # Helper functions
    deep_getattr,
    is_last_rank,
    load_hf_weights_to_megatron,
    # Legacy compatibility
    create_megatron_args,
    set_megatron_args,
    restore_megatron_args,
    mock_megatron_args,
)
from .initializer import MegatronModelInitializer, initialize_megatron_model
from .qwen3 import Qwen3ModelMeta, get_model_default_config

__all__ = [
    # Bridge classes
    'TwinkleBridgeAdapter',
    'TwinkleGPTBridge',
    'BridgeConfig',
    'SafetensorLoader',
    'StreamingSafetensorSaver',
    'LazyTensor',
    # Helper functions
    'deep_getattr',
    'is_last_rank',
    'load_hf_weights_to_megatron',
    # Legacy compatibility
    'create_megatron_args',
    'set_megatron_args',
    'restore_megatron_args',
    'mock_megatron_args',
    # Initializer
    'MegatronModelInitializer',
    'initialize_megatron_model',
    # Model metadata
    'Qwen3ModelMeta',
    'get_model_default_config',
]
