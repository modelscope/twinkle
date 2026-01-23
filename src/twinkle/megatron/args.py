# Copyright (c) twinkle authors. All rights reserved.
"""Twinkle Args - A lightweight args system compatible with Megatron's get_args().

This module provides a TwinkleArgs class that mimics Megatron's global args.

Usage:
    from twinkle.megatron.args import get_args, set_args, TwinkleArgs
    
    # Initialize args from HF config
    args = TwinkleArgs.from_hf_config(hf_config, model_dir='/path/to/model')
    set_args(args)
    
    # Later, get_args() will return the set args
    args = get_args()
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

# Global args storage
_GLOBAL_ARGS: Optional['TwinkleArgs'] = None


def get_args() -> 'TwinkleArgs':
    """Get the global TwinkleArgs instance.
    
    This function is designed to be a drop-in replacement for megatron's get_args().
    If TwinkleArgs has not been set, it will try to use megatron's get_args() as fallback.
    
    Returns:
        TwinkleArgs instance or megatron args.
        
    Raises:
        RuntimeError: If args have not been initialized.
    """
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is not None:
        return _GLOBAL_ARGS
    
    # Fallback to megatron's get_args if available
    try:
        from megatron.training import get_args as megatron_get_args
        return megatron_get_args()
    except (ImportError, AssertionError):
        pass
    
    raise RuntimeError(
        "Twinkle args have not been initialized. "
        "Please call set_args(TwinkleArgs.from_hf_config(...)) before using get_args()."
    )


def set_args(args: 'TwinkleArgs') -> None:
    """Set the global TwinkleArgs instance."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def clear_args() -> None:
    """Clear the global args."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None


@dataclass
class TwinkleArgs:
    """Lightweight args class compatible with Megatron's args.
    
    This class provides a unified configuration system for both model creation 
    and weight conversion. It stores a reference to the original HuggingFace config
    and implements __getattr__ to fallback to hf_config for missing attributes.
    
    Attributes:
        _hf_config: The original HuggingFace config object (stored but not a dataclass field).
    """
    # =========================================================================
    # Model architecture (from HF config)
    # =========================================================================
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    num_layers: int = 32
    ffn_hidden_size: int = 11008
    vocab_size: int = 32000
    padded_vocab_size: int = 32000
    kv_channels: Optional[int] = None  # head_dim
    
    # =========================================================================
    # Parallelism settings
    # =========================================================================
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    
    # =========================================================================
    # RoPE settings
    # =========================================================================
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    max_position_embeddings: int = 4096
    original_max_position_embeddings: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # =========================================================================
    # Model settings
    # =========================================================================
    model_dir: str = ''
    hf_model_type: str = 'qwen2'
    is_multimodal: bool = False
    
    # =========================================================================
    # Bias settings (used by bridge for weight conversion)
    # =========================================================================
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = False
    tie_word_embeddings: bool = False
    
    # =========================================================================
    # MoE settings (used by bridge for weight conversion)
    # =========================================================================
    num_experts: int = 0
    num_experts_per_tok: int = 2
    shared_expert_intermediate_size: int = 0
    
    # =========================================================================
    # Training/inference settings
    # =========================================================================
    torch_dtype: torch.dtype = torch.bfloat16
    task_type: str = 'causal_lm'
    num_labels: int = 2
    
    # =========================================================================
    # Attention settings
    # =========================================================================
    attn_impl: str = 'flash_attn'
    attention_backend: Any = None
    
    # =========================================================================
    # MTP (Multi-Token Prediction) settings
    # =========================================================================
    mtp_num_layers: int = 0
    
    # =========================================================================
    # MLA (Multi-Latent Attention) settings - for DeepSeek-V2/V3 style models
    # =========================================================================
    multi_latent_attention: bool = False
    q_lora_rank: Optional[int] = None
    
    # =========================================================================
    # LoRA/PEFT settings
    # =========================================================================
    merge_lora: bool = False
    target_modules: List[str] = field(default_factory=list)
    freeze_llm: bool = False
    freeze_vit: bool = False
    freeze_aligner: bool = False
    
    # =========================================================================
    # FP8 quantization settings
    # =========================================================================
    fp8: Optional[str] = None
    fp8_recipe: str = 'delayed'
    fp8_param_gather: bool = False
    
    # =========================================================================
    # Additional settings
    # =========================================================================
    untie_embeddings_and_output_weights: bool = True
    max_shard_size: str = '5GB'
    llm_model_type: str = 'gpt'  # For transformers 5.0 compatibility
    
    def __post_init__(self):
        # Initialize _hf_config as None (will be set by from_hf_config)
        object.__setattr__(self, '_hf_config', None)
        object.__setattr__(self, '_text_config', None)
        
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.kv_channels is None:
            self.kv_channels = self.hidden_size // self.num_attention_heads
        if self.attention_backend is None:
            self.attention_backend = SimpleNamespace(name='flash')
    
    def __getattr__(self, name: str) -> Any:
        """Fallback to hf_config for missing attributes.
        
        This allows seamless access to HuggingFace config attributes that
        weren't explicitly copied to TwinkleArgs.
        """
        # Avoid infinite recursion for special attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Try to get from hf_config
        hf_config = object.__getattribute__(self, '_hf_config')
        if hf_config is not None:
            # First try direct access
            if hasattr(hf_config, name):
                return getattr(hf_config, name)
            
            # For multimodal models, try text_config
            text_config = object.__getattribute__(self, '_text_config')
            if text_config is not None and hasattr(text_config, name):
                return getattr(text_config, name)
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}' "
            f"and it was not found in hf_config either."
        )
    
    # =========================================================================
    # Convenience property aliases
    # =========================================================================
    @property
    def tp_size(self) -> int:
        return self.tensor_model_parallel_size
    
    @property
    def pp_size(self) -> int:
        return self.pipeline_model_parallel_size
    
    @property
    def cp_size(self) -> int:
        return self.context_parallel_size
    
    @property
    def ep_size(self) -> int:
        return self.expert_model_parallel_size
    
    @property
    def etp_size(self) -> int:
        return self.expert_tensor_parallel_size
    
    @property
    def head_dim(self) -> int:
        return self.kv_channels
    
    @property
    def intermediate_size(self) -> int:
        return self.ffn_hidden_size
    
    @property
    def num_query_groups(self) -> int:
        """Alias for num_key_value_heads (Megatron naming)."""
        return self.num_key_value_heads
    
    @property
    def group_query_attention(self) -> bool:
        """Whether the model uses grouped query attention (GQA)."""
        return self.num_key_value_heads != self.num_attention_heads
    
    @property
    def hf_config(self) -> Any:
        """Get the original HuggingFace config."""
        return object.__getattribute__(self, '_hf_config')
    
    @property
    def text_config(self) -> Any:
        """Get the text config (for multimodal models)."""
        return object.__getattribute__(self, '_text_config')
    
    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        model_dir: str = '',
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        etp_size: Optional[int] = None,
        sequence_parallel: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        task_type: str = 'causal_lm',
        padded_vocab_size: Optional[int] = None,
    ) -> 'TwinkleArgs':
        """Create TwinkleArgs from a HuggingFace model config.
        
        This method handles both regular LLM configs and multimodal configs
        where parameters may be in nested sub-configs (e.g., text_config).
        
        The original hf_config is stored and can be accessed via args.hf_config
        or through attribute fallback (__getattr__).
        """
        # Handle multimodal configs with nested text_config
        text_config = hf_config
        if hasattr(hf_config, 'text_config') and hf_config.text_config is not None:
            text_config = hf_config.text_config
        
        vocab_size = getattr(text_config, 'vocab_size', 32000)
        if padded_vocab_size is None:
            divisor = tp_size * 128
            padded_vocab_size = ((vocab_size + divisor - 1) // divisor) * divisor
        
        num_attention_heads = getattr(text_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(text_config, 'num_key_value_heads', num_attention_heads)
        hidden_size = getattr(text_config, 'hidden_size', 4096)
        
        # Get kv_channels (head_dim)
        kv_channels = getattr(text_config, 'head_dim', None)
        if kv_channels is None:
            kv_channels = hidden_size // num_attention_heads
        
        # Get rope_scaling
        rope_scaling = getattr(text_config, 'rope_scaling', None)
        
        # Detect multimodal model
        model_type = getattr(hf_config, 'model_type', 'qwen2')
        is_multimodal = 'vl' in model_type.lower() or 'vision' in model_type.lower() or 'omni' in model_type.lower()
        
        # Determine QKV bias
        if hasattr(text_config, 'attention_bias'):
            add_qkv_bias = text_config.attention_bias
        elif model_type in ('qwen2', 'qwen2_5', 'qwen2_vl', 'qwen2_5_vl'):
            add_qkv_bias = True
        else:
            add_qkv_bias = False
        
        # Determine QK layernorm
        qk_layernorm = getattr(text_config, 'qk_layernorm', False) or \
                       getattr(text_config, 'use_qk_norm', False)
        if not qk_layernorm and model_type in ('qwen3', 'qwen3_moe', 'qwen3_vl', 'qwen3_vl_moe'):
            qk_layernorm = True
        
        # MoE config
        num_experts = getattr(text_config, 'num_experts', 0) or \
                      getattr(text_config, 'n_routed_experts', 0) or \
                      getattr(text_config, 'num_local_experts', 0) or 0
        num_experts_per_tok = getattr(text_config, 'num_experts_per_tok', 2) or \
                              getattr(text_config, 'moe_topk', 2) or 2
        shared_expert_size = getattr(text_config, 'shared_expert_intermediate_size', 0) or 0
        
        # MLA config (for DeepSeek-V2/V3 style models)
        q_lora_rank = getattr(text_config, 'q_lora_rank', None)
        multi_latent_attention = q_lora_rank is not None
        
        # Create instance
        instance = cls(
            # Model architecture
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_layers=getattr(text_config, 'num_hidden_layers', 32),
            ffn_hidden_size=getattr(text_config, 'intermediate_size', 11008),
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
            kv_channels=kv_channels,
            # Parallelism
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size or tp_size,
            sequence_parallel=sequence_parallel,
            # RoPE
            rotary_base=int(getattr(text_config, 'rope_theta', 10000)),
            rotary_percent=1.0,
            max_position_embeddings=getattr(text_config, 'max_position_embeddings', 4096),
            original_max_position_embeddings=getattr(text_config, 'original_max_position_embeddings', None),
            rope_scaling=rope_scaling,
            # Model settings
            model_dir=model_dir,
            hf_model_type=model_type,
            is_multimodal=is_multimodal,
            # Bias settings
            add_qkv_bias=add_qkv_bias,
            add_bias_linear=getattr(text_config, 'mlp_bias', False),
            qk_layernorm=qk_layernorm,
            tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings', False),
            # MoE settings
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate_size=shared_expert_size,
            # MLA settings
            multi_latent_attention=multi_latent_attention,
            q_lora_rank=q_lora_rank,
            # Training
            torch_dtype=torch_dtype,
            task_type=task_type,
            # Attention
            attn_impl='flash_attn',
            attention_backend=SimpleNamespace(name='flash'),
            # Other
            untie_embeddings_and_output_weights=not getattr(hf_config, 'tie_word_embeddings', False),
        )
        
        # Store the original hf_config for attribute fallback
        object.__setattr__(instance, '_hf_config', hf_config)
        object.__setattr__(instance, '_text_config', text_config if text_config is not hf_config else None)
        
        return instance