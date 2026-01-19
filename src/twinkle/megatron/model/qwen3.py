# Copyright (c) twinkle authors. All rights reserved.
"""Qwen3 model metadata for Megatron-Core.

This module provides metadata for Qwen3 models.
"""
from typing import Any, Dict


# =============================================================================
# Qwen3 Model Metadata
# =============================================================================
class Qwen3ModelMeta:
    """Metadata for Qwen3 models."""

    # Supported architectures
    DENSE_ARCHITECTURES = [
        'Qwen3ForCausalLM', 'Qwen2ForCausalLM', 'Qwen2.5ForCausalLM'
    ]
    MOE_ARCHITECTURES = ['Qwen3MoeForCausalLM', 'Qwen2MoeForCausalLM']
    ALL_ARCHITECTURES = DENSE_ARCHITECTURES + MOE_ARCHITECTURES

    # HuggingFace key prefixes
    HF_LAYERS_PREFIX = 'model.layers'
    HF_EMBED_KEY = 'model.embed_tokens.weight'
    HF_FINAL_LAYERNORM_KEY = 'model.norm.weight'
    HF_LM_HEAD_KEY = 'lm_head.weight'

    # Qwen3 specific settings
    DEFAULT_CONFIG = {
        'qk_layernorm': True,
        'swiglu': True,
        'disable_bias_linear': True,
        'rotary_interleaved': False,
    }

    # MoE specific settings
    MOE_CONFIG = {
        'use_shared_expert_gate': True,
    }

    @classmethod
    def is_qwen3(cls, architecture: str) -> bool:
        """Check if architecture is a Qwen3 model."""
        return architecture in cls.ALL_ARCHITECTURES

    @classmethod
    def is_qwen3_moe(cls, architecture: str) -> bool:
        """Check if architecture is a Qwen3 MoE model."""
        return architecture in cls.MOE_ARCHITECTURES


def get_model_default_config(architecture: str) -> Dict[str, Any]:
    """Get default config overrides for a model architecture.

    Args:
        architecture: Model architecture name.

    Returns:
        Default config dict for Megatron TransformerConfig.
    """
    if Qwen3ModelMeta.is_qwen3_moe(architecture):
        return {**Qwen3ModelMeta.DEFAULT_CONFIG, **Qwen3ModelMeta.MOE_CONFIG}
    elif Qwen3ModelMeta.is_qwen3(architecture):
        return Qwen3ModelMeta.DEFAULT_CONFIG
    return {}
