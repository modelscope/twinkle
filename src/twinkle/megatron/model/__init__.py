# Copyright (c) twinkle authors. All rights reserved.
from . import mm_gpts
from .constant import ModelType, MegatronModelType

# Model registration
from .register import MegatronModelMeta, register_megatron_model, get_megatron_model_meta

# Bridge classes (GPTBridge and MultimodalGPTBridge from swift-style implementation)
from .gpt_bridge import GPTBridge, MultimodalGPTBridge

# GPT model
from .gpt_model import GPTModel

# Multimodal model
from .mm_gpt_model import MultimodalGPTModel

# Bridge implementation (twinkle-specific, full implementation)
from .bridge import (
    TwinkleGPTBridge, TwinkleBridgeAdapter,
    SafetensorLazyLoader, StreamingSafetensorSaver,
)

# Initializer
from .initializer import MegatronModelInitializer

# Multi-tenant
from .multi_tenant_megatron import MegatronMultiAdapter, MultiTenantMegatronModel

# Qwen3-VL model classes (from mm_gpts)
from .mm_gpts.qwen3_vl import (
    Qwen3VLGPTModel, Qwen3VLTransformerBlock, 
    Qwen3VL_Vit, Qwen3Omni_Vit
)

# HuggingFaceModule utility
from .mm_gpts.utils import HuggingFaceModule
