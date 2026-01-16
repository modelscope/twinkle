# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core integration for twinkle training framework.

This module provides independent implementation for Megatron support,
"""

from .model import (BridgeConfig, LazyTensor, MegatronModelInitializer,
                    Qwen3ModelMeta, SafetensorLoader, StreamingSafetensorSaver,
                    TwinkleBridgeAdapter, TwinkleGPTBridge,
                    create_megatron_args, get_model_default_config,
                    initialize_megatron_model, is_last_rank,
                    load_hf_weights_to_megatron, mock_megatron_args,
                    restore_megatron_args, set_megatron_args)
from .tuners import LoraParallelLinear, dispatch_megatron

from .model import deep_getattr as bridge_deep_getattr  # Bridge classes; Helper functions; Avoid conflict with utils.deep_getattr; Legacy compatibility; Initializer; Qwen3 support
from .utils import (  # Layer finding; Model preparation; Config conversion; Utilities; Multi-tenant support; Training state
    MegatronTrainerState, TenantProcessGroupManager, convert_hf_config,
    deep_getattr, find_all_linears, find_embedding, find_router,
    forward_step_helper, get_model_parameter_info, get_padding_to,
    get_target_modules, get_tenant_manager, patch_deepcopy, prepare_lora_model,
    prepare_mcore_model, set_linear_is_expert, tuners_sharded_state_dict)

__all__ = [
    # Tuners
    'LoraParallelLinear',
    'dispatch_megatron',
    # Layer finding
    'find_all_linears',
    'find_router',
    'find_embedding',
    'get_target_modules',
    'set_linear_is_expert',
    # Model preparation
    'prepare_mcore_model',
    'prepare_lora_model',
    # Config conversion
    'convert_hf_config',
    # Utilities
    'get_model_parameter_info',
    'get_padding_to',
    'patch_deepcopy',
    'tuners_sharded_state_dict',
    'forward_step_helper',
    'deep_getattr',
    # Multi-tenant support
    'TenantProcessGroupManager',
    'get_tenant_manager',
    # Training state
    'MegatronTrainerState',
    # Bridge classes
    'TwinkleBridgeAdapter',
    'TwinkleGPTBridge',
    'BridgeConfig',
    'SafetensorLoader',
    'StreamingSafetensorSaver',
    'LazyTensor',
    # Helper functions
    'load_hf_weights_to_megatron',
    'is_last_rank',
    # Legacy compatibility
    'create_megatron_args',
    'set_megatron_args',
    'restore_megatron_args',
    'mock_megatron_args',
    # Initializer
    'MegatronModelInitializer',
    'initialize_megatron_model',
    # Qwen3 support
    'Qwen3ModelMeta',
    'get_model_default_config',
]
