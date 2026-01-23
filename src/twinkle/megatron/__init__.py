# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core integration for twinkle training framework.

This module provides independent implementation for Megatron support.
"""

# Args system (compatible with megatron's get_args())
from .args import TwinkleArgs, get_args, set_args, clear_args

# Model type constants and registration
from .model import (
    ModelType, MegatronModelType,
    # Registration
    MegatronModelMeta, register_megatron_model, get_megatron_model_meta,
    TwinkleBridgeAdapter, TwinkleGPTBridge,
    SafetensorLazyLoader, StreamingSafetensorSaver,
    # Model classes
    GPTModel, MultimodalGPTModel,
    # Initializer
    MegatronModelInitializer,
    # Vision module utility
    HuggingFaceModule,
)

from .tuners import LoraParallelLinear, dispatch_megatron