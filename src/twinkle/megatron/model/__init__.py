# Copyright (c) twinkle authors. All rights reserved.
"""Megatron model initialization and weight conversion.

This module provides independent implementation for weight loading/saving,
and multi-tenant model wrapper for LoRA training.
"""

from .bridge import (  # Main classes; Helper functions; Legacy compatibility
    BridgeConfig, LazyTensor, SafetensorLoader, StreamingSafetensorSaver,
    TwinkleBridgeAdapter, TwinkleBridgeInitializer, TwinkleGPTBridge,
    create_megatron_args, deep_getattr, is_last_rank,
    load_hf_weights_to_megatron, mock_megatron_args, restore_megatron_args,
    set_megatron_args)
from .initializer import MegatronModelInitializer, initialize_megatron_model
from .multi_tenant_megatron import (MegatronMultiAdapter,
                                    MultiTenantMegatronModel)
from .qwen3 import Qwen3ModelMeta, get_model_default_config

__all__ = [
    # Bridge classes
    'TwinkleBridgeAdapter',
    'TwinkleBridgeInitializer',
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
    # Multi-tenant
    'MultiTenantMegatronModel',
    'MegatronMultiAdapter',
]
