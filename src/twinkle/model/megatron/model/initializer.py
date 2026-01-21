# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron model initialization from HuggingFace checkpoints."""
from dataclasses import fields
from typing import Any, Optional

import torch
import torch.nn as nn

from ..utils import convert_hf_config


def _get_transformer_config_fields() -> set:
    """Get valid field names for TransformerConfig.

    Returns:
        Set of valid field names.
    """
    from megatron.core.transformer import TransformerConfig
    return {f.name for f in fields(TransformerConfig)}


class MegatronModelInitializer:
    """Initialize Megatron-Core models from HuggingFace checkpoints.

    This class handles:
    - Converting HuggingFace config to Megatron TransformerConfig
    - Creating Megatron model architecture
    - Loading HuggingFace weights into Megatron model
    """
    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        etp_size: Optional[int] = None,
        vp_size: Optional[int] = None,
        sequence_parallel: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
        use_cpu_initialization: bool = True,
    ):
        """Initialize MegatronModelInitializer.

        Args:
            tp_size: Tensor parallel size.
            pp_size: Pipeline parallel size.
            cp_size: Context parallel size.
            ep_size: Expert parallel size.
            etp_size: Expert tensor parallel size (defaults to tp_size).
            vp_size: Virtual pipeline parallel size.
            sequence_parallel: Enable sequence parallelism.
            params_dtype: Parameter data type.
            use_cpu_initialization: Initialize model on CPU first.
        """
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.ep_size = ep_size
        self.etp_size = etp_size or tp_size
        self.vp_size = vp_size
        self.sequence_parallel = sequence_parallel
        self.params_dtype = params_dtype
        self.use_cpu_initialization = use_cpu_initialization

        # Cache valid TransformerConfig fields
        self._valid_config_fields = _get_transformer_config_fields()

    def create_transformer_config(
        self,
        hf_config: Any,
        **overrides,
    ) -> 'TransformerConfig':
        """Create Megatron TransformerConfig from HuggingFace config.

        Args:
            hf_config: HuggingFace model config.
            **overrides: Config overrides.

        Returns:
            Megatron TransformerConfig.
        """
        from megatron.core.transformer import TransformerConfig
        # Convert HuggingFace config to dict
        mg_config_dict = convert_hf_config(hf_config)

        # Apply overrides
        mg_config_dict.update(overrides)

        # Build config kwargs with only valid fields
        config_kwargs = {
            # Required fields
            'num_layers': mg_config_dict['num_layers'],
            'hidden_size': mg_config_dict['hidden_size'],
            'num_attention_heads': mg_config_dict['num_attention_heads'],
            # Parallel settings
            'tensor_model_parallel_size': self.tp_size,
            'pipeline_model_parallel_size': self.pp_size,
            'context_parallel_size': self.cp_size,
            'expert_model_parallel_size': self.ep_size,
            'sequence_parallel': self.sequence_parallel,
            'params_dtype': self.params_dtype,
            'use_cpu_initialization': self.use_cpu_initialization,
        }

        # Optional fields - only add if valid for this Megatron version
        optional_fields = {
            'num_query_groups':
            mg_config_dict.get('num_query_groups',
                               mg_config_dict['num_attention_heads']),
            'ffn_hidden_size':
            mg_config_dict.get('ffn_hidden_size',
                               4 * mg_config_dict['hidden_size']),
            'num_moe_experts':
            mg_config_dict.get('num_experts'),
            'moe_router_topk':
            mg_config_dict.get('moe_router_topk', 2)
            if mg_config_dict.get('num_experts') else None,
            'layernorm_epsilon':
            mg_config_dict.get('norm_epsilon', 1e-6),
            'add_qkv_bias':
            mg_config_dict.get('add_qkv_bias', False),
            'add_bias_linear':
            not mg_config_dict.get('disable_bias_linear', True),
            'gated_linear_unit':
            mg_config_dict.get('swiglu', True),
            'qk_layernorm':
            mg_config_dict.get('qk_layernorm', False),
            'normalization':
            'RMSNorm',
        }

        # Add optional fields that are valid for this Megatron version
        for key, value in optional_fields.items():
            if key in self._valid_config_fields and value is not None:
                config_kwargs[key] = value

        # Store rotary settings for GPTModel (not TransformerConfig)
        self._rotary_base = mg_config_dict.get('rotary_base', 10000)
        self._rotary_percent = mg_config_dict.get('rotary_percent', 1.0)
        self._position_embedding_type = mg_config_dict.get(
            'position_embedding_type', 'rope')

        # Create TransformerConfig
        config = TransformerConfig(**config_kwargs)

        return config

    def create_gpt_model(
        self,
        hf_config: Any,
        vocab_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        **config_overrides,
    ) -> 'GPTModel':
        """Create Megatron GPT model from HuggingFace config.

        Args:
            hf_config: HuggingFace model config.
            vocab_size: Override vocab size.
            max_sequence_length: Override max sequence length.
            **config_overrides: Config overrides.

        Returns:
            Megatron GPTModel.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.models.gpt import GPTModel
        # Create config (also sets self._rotary_base, etc.)
        config = self.create_transformer_config(hf_config, **config_overrides)

        # Get vocab size
        if vocab_size is None:
            vocab_size = hf_config.vocab_size

        # Pad vocab size for tensor parallelism
        padded_vocab_size = self._pad_vocab_size(vocab_size)

        # Get max sequence length
        if max_sequence_length is None:
            max_sequence_length = getattr(hf_config, 'max_position_embeddings',
                                          4096)

        # Get tie_word_embeddings setting
        tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        # Create model with rotary settings passed directly to GPTModel
        model = GPTModel(
            config=config,
            transformer_layer_spec=self._get_layer_spec(config),
            vocab_size=padded_vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
            parallel_output=True,
            share_embeddings_and_output_weights=tie_word_embeddings,
            position_embedding_type=self._position_embedding_type,
            rotary_percent=self._rotary_percent,
            rotary_base=self._rotary_base,
        )

        return model

    def _pad_vocab_size(self, vocab_size: int) -> int:
        """Pad vocab size for tensor parallelism.

        Args:
            vocab_size: Original vocab size.

        Returns:
            Padded vocab size.
        """
        # Pad to multiple of tp_size * 128 for efficient parallelism
        divisor = self.tp_size * 128
        return ((vocab_size + divisor - 1) // divisor) * divisor

    def _get_layer_spec(self, config: 'TransformerConfig'):
        """Get transformer layer specification.

        Args:
            config: Transformer config.

        Returns:
            Layer specification (ModuleSpec or TransformerBlockSubmodules).
        """
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
            get_gpt_layer_local_spec,
        )

        # Determine if this is a MoE model
        num_experts = getattr(config, 'num_moe_experts', None)
        moe_grouped_gemm = getattr(config, 'moe_grouped_gemm', False)
        qk_layernorm = getattr(config, 'qk_layernorm', False)
        multi_latent_attention = getattr(config, 'multi_latent_attention',
                                         False)

        # Try TE (TransformerEngine) layers first for better performance
        try:
            return get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_experts,
                moe_grouped_gemm=moe_grouped_gemm,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=multi_latent_attention,
            )
        except (ImportError, AttributeError):
            raise RuntimeError("TransformerEngine is not installed or not compatible with this version of Megatron-Core.")

    def load_from_hf(
        self,
        model: nn.Module,
        hf_model_path: str,
        hf_config: Any,
    ) -> None:
        """Load HuggingFace checkpoint into Megatron model.

        Args:
            model: The Megatron model.
            hf_model_path: Path to HuggingFace checkpoint or model ID.
            hf_config: HuggingFace model config.
        """
        import os

        # Resolve model path if it's a model ID (not a local path)
        if not os.path.isdir(hf_model_path):
            from twinkle.hub import HubOperation
            hf_model_path = HubOperation.download_model(hf_model_path)

        # Calculate padded vocab size
        padded_vocab_size = self._pad_vocab_size(hf_config.vocab_size)

        # Use BridgeAdapter
        from .bridge import BridgeAdapter
        adapter = BridgeAdapter(
            hf_config=hf_config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            ep_size=self.ep_size,
            model_path=hf_model_path,
            padded_vocab_size=padded_vocab_size,
        )
        adapter.load_weights(model, hf_model_path)


def initialize_megatron_model(
    hf_model_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    params_dtype: torch.dtype = torch.bfloat16,
    load_weights: bool = True,
) -> nn.Module:
    """Convenience function to initialize Megatron model from HuggingFace checkpoint.

    Args:
        hf_model_path: Path to HuggingFace checkpoint.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        ep_size: Expert parallel size.
        params_dtype: Parameter data type.
        load_weights: Whether to load weights.

    Returns:
        Initialized Megatron model.
    """
    from transformers import AutoConfig

    # Load HuggingFace config
    hf_config = AutoConfig.from_pretrained(hf_model_path)

    # Create initializer
    initializer = MegatronModelInitializer(
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        params_dtype=params_dtype,
    )

    # Create model
    model = initializer.create_gpt_model(hf_config)

    # Load weights
    if load_weights:
        initializer.load_from_hf(model, hf_model_path, hf_config)

    return model
