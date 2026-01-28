# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Literal
import inspect
import torch
import torch.nn as nn

# Global args storage
_GLOBAL_ARGS: Optional['TwinkleMegatronArgs'] = None

from twinkle import DeviceMesh
from twinkle.utils import exists
from .utils import convert_hf_config


def get_args() -> 'TwinkleMegatronArgs':
    """Get the global TwinkleMegatronArgs instance.
    
    This function is designed to be a drop-in replacement for megatron's get_args().
    If TwinkleMegatronArgs has not been set, it will try to use megatron's get_args() as fallback.
    
    Returns:
        TwinkleMegatronArgs instance or megatron args.
        
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
    )


def set_args(args: 'TwinkleMegatronArgs') -> None:
    """Set the global TwinkleMegatronArgs instance."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def clear_args() -> None:
    """Clear the global args."""
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None


@dataclass
class TwinkleMegatronArgs:
    """Lightweight args class compatible with Megatron's args.
    
    This class provides a unified configuration system for both model creation 
    and weight conversion. It stores a reference to the original HuggingFace config
    and implements __getattr__ to fallback to hf_config for missing attributes.
    
    Attributes:
        _hf_config: The original HuggingFace config object (stored but not a dataclass field).
    """
    _model: Optional[List[nn.Module]] = None
    # =========================================================================
    # Model architecture (from HF config)
    # =========================================================================
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    num_layers: int = 32
    ffn_hidden_size: int = 11008
    vocab_size: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    kv_channels: Optional[int] = None  # head_dim
    
    # =========================================================================
    # Parallelism settings
    # =========================================================================
    device_mesh: DeviceMesh = None
    sequence_parallel: bool = False
    
    # =========================================================================
    # RoPE settings
    # =========================================================================
    rotary_base: int = 10000  # rope_theta in HF config
    rotary_percent: float = 1.0
    max_position_embeddings: int = 4096
    original_max_position_embeddings: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: Optional[float] = None  # For partial RoPE
    rope_interleaved: bool = False  # mrope_interleaved in Swift
    
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
    params_dtype: torch.dtype = torch.bfloat16
    task_type: str = 'causal_lm' # not used for now
    num_labels: int = 2
    
    # =========================================================================
    # Attention settings
    # =========================================================================
    attn_impl: str = 'flash_attn'
    attention_backend: str = 'flash'
    
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
    # Activation checkpointing settings
    # =========================================================================
    recompute_granularity: Literal['selective', 'full', 'none'] = 'selective'
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])
    recompute_method: Optional[Literal['uniform', 'block']] = None
    recompute_num_layers: Optional[int] = None
    # =========================================================================
    # Additional settings
    # =========================================================================
    untie_embeddings_and_output_weights: bool = True
    max_shard_size: str = '5GB'
    llm_model_type: str = 'gpt'  # For transformers 5.0 compatibility
    use_cpu_initialization: bool = False
    
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
        weren't explicitly copied to TwinkleMegatronArgs.
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

    @property
    def tensor_model_parallel_size(self) -> int:
        return self.device_mesh.tp_world_size

    @property
    def tp_size(self) -> int:
        return self.device_mesh.tp_world_size
    
    @property
    def pipeline_model_parallel_size(self) -> int:
        return self.device_mesh.pp_world_size

    @property
    def pp_size(self) -> int:
        return self.device_mesh.pp_world_size
    
    @property
    def context_parallel_size(self) -> int:
        return self.device_mesh.cp_world_size or 1

    @property
    def cp_size(self) -> int:
        return self.device_mesh.cp_world_size or 1
    
    @property
    def expert_model_parallel_size(self) -> int:
        return self.device_mesh.ep_size or 1

    @property
    def ep_size(self) -> int:
        return self.device_mesh.ep_size
    
    @property
    def expert_tensor_parallel_size(self) -> int:
        return self.device_mesh.etp_world_size

    @property
    def etp_size(self) -> int:
        return self.expert_tensor_parallel_size
    
    @property
    def virtual_pipeline_model_parallel_size(self) -> int:
        return self.device_mesh.vpp_size

    @property
    def vpp_size(self) -> int:
        return self.device_mesh.vpp_size

    @property
    def order(self) -> str:
        return self.device_mesh.order
    
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
    def torch_dtype(self) -> torch.dtype:
        return self.params_dtype

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
        device_mesh: DeviceMesh = None,
        params_dtype: torch.dtype = torch.bfloat16,
        sequence_parallel: bool = False,
        task_type: str = 'causal_lm',
        padded_vocab_size: Optional[int] = None,
        **kwargs,
    ) -> 'TwinkleMegatronArgs':
        """Create TwinkleMegatronArgs from a HuggingFace model config.
        
        This method handles both regular LLM configs and multimodal configs
        where parameters may be in nested sub-configs (e.g., text_config).
        
        The original hf_config is stored and can be accessed via args.hf_config
        or through attribute fallback (__getattr__).
        """
        # Handle multimodal configs with nested text_config
        text_config = hf_config
        if hasattr(hf_config, 'text_config') and hf_config.text_config is not None:
            text_config = hf_config.text_config
        
        vocab_size = getattr(text_config, 'vocab_size')
        assert vocab_size is not None, 'detect vocab_size in hf config failed'
        if padded_vocab_size is None:
            divisor = device_mesh.tp_world_size * 128
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
            device_mesh=device_mesh,
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
            params_dtype=params_dtype,
            task_type=task_type,
            # Attention
            attn_impl='flash_attn',
            attention_backend='flash',
            # Other
            untie_embeddings_and_output_weights=not getattr(hf_config, 'tie_word_embeddings', False),
            **kwargs,
        )
        
        # Store the original hf_config for attribute fallback
        object.__setattr__(instance, '_hf_config', hf_config)
        object.__setattr__(instance, '_text_config', text_config if text_config is not hf_config else None)
        
        # Apply convert_hf_config results to instance (like swift's init_model_args)
        # This ensures derived values like qk_layernorm are correctly set
        mg_config = convert_hf_config(hf_config)
        for k, v in mg_config.items():
            if not hasattr(instance, k):
                continue
            current_value = getattr(instance, k)
            if current_value is None:
                object.__setattr__(instance, k, v)
            elif current_value is False and isinstance(v, bool) and v:
                # update false
                object.__setattr__(instance, k, v)
        
        return instance


    def create_model(
        self,
    ) -> List[nn.Module]:
        """Create Megatron GPT model from HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration.
            padded_vocab_size: Padded vocabulary size.

        Returns:
            Megatron GPT model.
        """
        if self._model is not None:
            return self._model
        from megatron.core import parallel_state as mpu
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.enums import AttnBackend
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec, )
        from .model.register import get_megatron_model_meta
        hf_config = self.hf_config
        padded_vocab_size = self.padded_vocab_size
        # Convert HF config to Megatron config
        mg_config_dict = convert_hf_config(hf_config)
        
        # Get registered model class (for multimodal models like Qwen3-VL)
        model_meta = get_megatron_model_meta(self.hf_model_type)
        ModelClass = model_meta.model_cls if model_meta else GPTModel

        # Build TransformerConfig
        num_attention_heads = mg_config_dict['num_attention_heads']
        num_query_groups = mg_config_dict.get('num_query_groups',
                                              num_attention_heads)
        num_layers = mg_config_dict['num_layers']

        # Configure activation recomputation
        recompute_method = self.recompute_method
        recompute_num_layers = self.recompute_num_layers

        # Auto-configure for 'full' recomputation if not specified
        if self.recompute_granularity == 'full':
            if recompute_method is None:
                recompute_method = 'uniform'
            if recompute_num_layers is None:
                # Recompute all layers for maximum memory savings
                recompute_num_layers = num_layers // self.pp_size

        # Create finalize_model_grads function for DP gradient synchronization
        # Megatron's native finalize_model_grads requires DDP-wrapped models with ddp_config.
        # For PEFT/LoRA models, we use a custom implementation that handles non-DDP models.
        from megatron.core.distributed import finalize_model_grads as _native_finalize_model_grads

        def finalize_model_grads_for_lora(model,
                                          num_tokens=None,
                                          pg_collection=None):
            from peft import PeftModel as _PeftModel
            from megatron.core.distributed import DistributedDataParallel as MegatronDDP
            
            # Check if model is DDP-wrapped (has ddp_config)
            # Need to unwrap PeftModel to check the underlying model
            def _get_base_model(m):
                if isinstance(m, _PeftModel):
                    return _get_base_model(m.base_model.model)
                return m
            
            base_model = _get_base_model(model[0])
            if isinstance(base_model, MegatronDDP) or hasattr(base_model, 'ddp_config'):
                # Use native implementation for DDP models
                return _native_finalize_model_grads(model, num_tokens,
                                                    pg_collection)

            return

        # MoE configuration
        num_experts = mg_config_dict.get('num_experts', 0) or 0
        moe_ffn_hidden_size = mg_config_dict.get('moe_ffn_hidden_size')
        moe_router_topk = mg_config_dict.get('moe_router_topk', 2) or 2
        moe_shared_expert_intermediate_size = mg_config_dict.get(
            'moe_shared_expert_intermediate_size')

        # Build MoE-related kwargs
        moe_kwargs = {}
        if num_experts > 0:
            moe_kwargs.update({
                'num_moe_experts':
                num_experts,
                'moe_router_topk':
                moe_router_topk,
                'moe_router_load_balancing_type':
                mg_config_dict.get('moe_router_load_balancing_type',
                                   'aux_loss'),
                # MoE performance optimizations
                'moe_token_dispatcher_type':
                mg_config_dict.get(
                    'moe_token_dispatcher_type', 'alltoall'
                ),  # 'alltoall' is more efficient than 'allgather'
                'moe_grouped_gemm':
                mg_config_dict.get(
                    'moe_grouped_gemm', True
                ),  # Enable for better performance (requires grouped_gemm package)
                'moe_aux_loss_coeff':
                mg_config_dict.get(
                    'moe_aux_loss_coeff',
                    0.0),  # Auxiliary load balancing loss coefficient
            })

            # FFN hidden size for MoE
            if moe_ffn_hidden_size:
                moe_kwargs['moe_ffn_hidden_size'] = moe_ffn_hidden_size

            # Shared expert configuration
            if moe_shared_expert_intermediate_size:
                moe_kwargs[
                    'moe_shared_expert_intermediate_size'] = moe_shared_expert_intermediate_size

            # Router score function (sigmoid for Qwen3, softmax for others)
            if mg_config_dict.get('moe_router_score_function'):
                moe_kwargs['moe_router_score_function'] = mg_config_dict[
                    'moe_router_score_function']

            # Expert bias for sigmoid router
            if mg_config_dict.get('moe_router_enable_expert_bias'):
                moe_kwargs['moe_router_enable_expert_bias'] = mg_config_dict[
                    'moe_router_enable_expert_bias']

        # Sequence parallel requires TP > 1
        # Auto-enable for MoE with TP > 1 (required by Megatron)
        use_sequence_parallel = self.sequence_parallel and self.tp_size > 1
        if num_experts > 0 and self.tp_size > 1 and not use_sequence_parallel:
            use_sequence_parallel = True
            print(
                f'Auto-enabling sequence_parallel for MoE with TP={self.tp_size}'
            )

        # For MoE models, ffn_hidden_size should be moe_ffn_hidden_size if not specified
        ffn_hidden_size = mg_config_dict.get('ffn_hidden_size')
        if ffn_hidden_size is None:
            ffn_hidden_size = moe_ffn_hidden_size or (
                4 * mg_config_dict['hidden_size'])

        # For models with non-standard head dimensions (like Qwen3-30B-A3B)
        kv_channels = mg_config_dict.get('kv_channels')

        # Activation function for SwiGLU (required by Megatron when gated_linear_unit=True)
        use_swiglu = mg_config_dict.get('swiglu', True)
        activation_func = torch.nn.functional.silu if use_swiglu else torch.nn.functional.gelu

        # Enable bias_activation_fusion for SwiGLU
        # Note: Only works with TransformerEngine and no bias in linear layers
        has_bias = not mg_config_dict.get('disable_bias_linear', True)
        bias_activation_fusion = use_swiglu and not has_bias

        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=mg_config_dict['hidden_size'],
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            context_parallel_size=self.cp_size,
            expert_model_parallel_size=self.ep_size or 1,
            virtual_pipeline_model_parallel_size=self.vpp_size,
            sequence_parallel=use_sequence_parallel,
            params_dtype=self.params_dtype,
            fp16=self.params_dtype == torch.float16,
            bf16=self.params_dtype == torch.bfloat16,
            pipeline_dtype=self.params_dtype,  # Required when using pipeline parallelism
            use_cpu_initialization=self.use_cpu_initialization,
            add_qkv_bias=self.add_qkv_bias,
            add_bias_linear=not mg_config_dict.get('disable_bias_linear',
                                                   True),
            gated_linear_unit=use_swiglu,
            activation_func=activation_func,  # SiLU for SwiGLU, GELU otherwise
            bias_activation_fusion=
            bias_activation_fusion,  # Fused SwiGLU for performance
            normalization='RMSNorm',
            layernorm_epsilon=mg_config_dict.get('norm_epsilon', 1e-6),
            qk_layernorm=mg_config_dict.get('qk_layernorm', False),
            hidden_dropout=0.0,
            attention_dropout=0.0,
            # Performance optimizations
            masked_softmax_fusion=True,  # Fused attention softmax
            bias_dropout_fusion=True,  # Fused bias + dropout
            apply_rope_fusion=True,  # Fused RoPE application
            attention_softmax_in_fp32=True,  # Numerical stability
            attention_backend=AttnBackend.flash,  # FlashAttention for speed
            # Activation recomputation for memory efficiency
            recompute_granularity=self.recompute_granularity,
            recompute_modules=self.recompute_modules
            if self.recompute_granularity == 'selective' else None,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            # Critical: Set finalize_model_grads_func for DP gradient synchronization
            # Uses custom wrapper that handles both DDP and PEFT/LoRA models
            finalize_model_grads_func=finalize_model_grads_for_lora,
            # MoE configuration
            **moe_kwargs,
        )
        if exists('megatron_core>=0.13'):
            config.expert_tensor_parallel_size = self.etp_size

        # Save transformer config for later use (e.g., DDP wrapping)
        self.config = config

        # Get layer spec - enable moe_grouped_gemm for MoE models
        moe_grouped_gemm = num_experts > 0
        try:
            layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=mg_config_dict.get('num_experts'),
                moe_grouped_gemm=moe_grouped_gemm,
                qk_layernorm=mg_config_dict.get('qk_layernorm', False),
            )
        except (ImportError, AttributeError):
            raise RuntimeError("TransformerEngine is not installed or not compatible with this version of Megatron-Core.")

        # Create model
        max_seq_length = getattr(hf_config, 'max_position_embeddings', 4096)
        rotary_base = mg_config_dict.get('rotary_base', 10000)
        extra_init_args = {}
        if hasattr(hf_config, 'rope_scaling') and hf_config.rope_scaling is not None and 'factor' in hf_config.rope_scaling:
            extra_init_args = {
                'seq_len_interpolation_factor': hf_config.rope_scaling["factor"]
            }
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            model = []
            has_vp_stage = inspect.signature(mpu.is_pipeline_first_stage).parameters.get("vp_stage", None) is not None
            for i in range(vpp_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                extra_kwargs = {} if not has_vp_stage else {"ignore_virtual": False, "vp_stage": i}
                if has_vp_stage:
                    extra_init_args['vp_stage'] = i
                _model = ModelClass(
                    config=config,
                    transformer_layer_spec=layer_spec,
                    vocab_size=padded_vocab_size,
                    max_sequence_length=max_seq_length,
                    pre_process=mpu.is_pipeline_first_stage(**extra_kwargs),
                    post_process=mpu.is_pipeline_last_stage(**extra_kwargs),
                    parallel_output=True,
                    share_embeddings_and_output_weights=getattr(
                        hf_config, 'tie_word_embeddings', False),
                    position_embedding_type='rope',
                    rotary_base=rotary_base,
                    **extra_init_args
                )
                model.append(_model)
            mpu.set_virtual_pipeline_model_parallel_rank(0)
        else:
            model = ModelClass(
                config=config,
                transformer_layer_spec=layer_spec,
                vocab_size=padded_vocab_size,
                max_sequence_length=max_seq_length,
                pre_process=mpu.is_pipeline_first_stage(),
                post_process=mpu.is_pipeline_last_stage(),
                parallel_output=True,
                share_embeddings_and_output_weights=getattr(
                    hf_config, 'tie_word_embeddings', False),
                position_embedding_type='rope',
                rotary_base=rotary_base,
                **extra_init_args,
            )
            model = [model]
        self._model = model
        return model
