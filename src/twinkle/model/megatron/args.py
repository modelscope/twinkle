# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional

from twinkle import DeviceMesh, Platform, get_logger
from twinkle.utils import exists
from .utils import convert_hf_config

# Global args storage
_GLOBAL_ARGS: Optional['TwinkleMegatronArgs'] = None
logger = get_logger()


def _normalize_word_embedding_allreduce_call(*call_args, **call_kwargs):
    """Normalize Megatron's private word-embedding helper call.

    Megatron Core has changed the helper signature across releases:
    - 0.12.1: (model, config)
    - 0.16.1: (model, config, embd_group, pp_group)
    - future releases may add more positional/keyword args.

    We keep the semantics stable and only normalize the known pieces.
    """
    model = call_kwargs.pop('model', call_args[0] if call_args else None)
    config = call_kwargs.pop('config', call_args[1] if len(call_args) > 1 else None)
    if model is None or config is None:
        raise TypeError('word-embedding finalize helper requires at least model and config arguments.')

    embd_group = call_kwargs.pop('embd_group', call_args[2] if len(call_args) > 2 else None)
    pp_group = call_kwargs.pop('pp_group', call_args[3] if len(call_args) > 3 else None)
    return model, config, embd_group, pp_group, call_kwargs


def _allreduce_word_embedding_grads_allow_none(*call_args, **call_kwargs):
    """None-safe drop-in for Megatron's private embedding all-reduce helper.

    This wrapper intentionally accepts arbitrary positional/keyword arguments so
    it can survive Megatron helper signature drift across versions.
    """
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import (_get_main_grad_attr, _reshard_if_dtensor,
                                                                _unshard_if_dtensor, get_attr_wrapped_model)

    model, config, embd_group, pp_group, _ = _normalize_word_embedding_allreduce_call(*call_args, **call_kwargs)
    if embd_group is None:
        embd_group = parallel_state.get_embedding_group()
    if pp_group is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()

    def _get_main_grad_attr_compat(weight, ddp_config):
        try:
            helper_params = inspect.signature(_get_main_grad_attr).parameters
        except (TypeError, ValueError):
            helper_params = None

        if helper_params is not None and len(helper_params) <= 1:
            return _get_main_grad_attr(weight)
        return _get_main_grad_attr(weight, ddp_config.use_custom_fsdp)

    if parallel_state.is_rank_in_embedding_group(
            ignore_virtual=True) and torch.distributed.get_world_size(embd_group) > 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:
            model_module = model[0]

        ddp_config = model_module.ddp_config
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)

        if model_module.share_embeddings_and_output_weights or getattr(config, 'mtp_num_layers', 0):
            weight = model_module.shared_embedding_or_output_weight()
            if weight is None:
                logger.warning_once(
                    'Megatron LoRA finalize skipped shared embedding/output weight all-reduce '
                    'because the tied weight is missing on this pipeline stage.',
                    hash_id='megatron_lora_skip_embedding_allreduce_missing_weight',
                )
                return

            grad_attr = _get_main_grad_attr_compat(weight, ddp_config)
            orig_grad = getattr(weight, grad_attr, None)
            grad = _unshard_if_dtensor(orig_grad)
            if grad is None:
                logger.warning_once(
                    'Megatron LoRA finalize skipped shared embedding/output weight all-reduce '
                    'because the tied weight has no grad. This is expected when LoRA freezes '
                    'the base embedding/output weight.',
                    hash_id='megatron_lora_skip_embedding_allreduce_none_grad',
                )
                return

            torch.distributed.all_reduce(grad, group=embd_group)
            setattr(weight, grad_attr, _reshard_if_dtensor(grad, orig_grad))


def get_args() -> 'TwinkleMegatronArgs':
    """Get the global TwinkleMegatronArgs instance.

    This function is designed to be a drop-in replacement for megatron's get_args().
    If TwinkleMegatronArgs has not been set, it will try to use megatron's get_args() as fallback.

    Returns:
        TwinkleMegatronArgs instance or megatron args.

    Raises:
        RuntimeError: If args have not been initialized.
    """
    if _GLOBAL_ARGS is not None:
        return _GLOBAL_ARGS

    raise RuntimeError('Twinkle args have not been initialized. ')


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
    variable_seq_lengths: bool = True

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
    moe_router_enable_expert_bias: bool = False

    # =========================================================================
    # Training/inference settings
    # =========================================================================
    params_dtype: torch.dtype = torch.bfloat16
    task_type: str = 'causal_lm'  # not used for now
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

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' "
                             f'and it was not found in hf_config either.')

    @property
    def tensor_model_parallel_size(self) -> int:
        return self.device_mesh.tp_world_size or 1

    @property
    def tp_size(self) -> int:
        return self.device_mesh.tp_world_size or 1

    @property
    def pipeline_model_parallel_size(self) -> int:
        return self.device_mesh.pp_world_size or 1

    @property
    def pp_size(self) -> int:
        return self.device_mesh.pp_world_size or 1

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
        return self.device_mesh.ep_size or 1

    @property
    def expert_tensor_parallel_size(self) -> int:
        if not exists('megatron_core>=0.13'):
            # megatron_core<0.13 does not have a separate ETP config. For expert ColumnParallelLinear,
            # the internal path still uses the dense TP group, and parameter sharding is determined by tp_size.
            # etp_size has no practical effect here.
            # Force alignment here to avoid a mismatch where GPTBridge shards by etp_size while
            # the parameters were built according to tp_size.
            tp = self.device_mesh.tp_world_size or 1
            if self.device_mesh.etp_size is not None and self.device_mesh.etp_world_size != tp:
                logger.warning(f'etp_size={self.device_mesh.etp_world_size} is ignored on '
                               f'megatron_core<0.13; expert TP is tied to tp_size={tp}')
            return tp
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
    def moe_shared_expert_intermediate_size(self) -> int:
        return self.shared_expert_intermediate_size

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
            if device_mesh.tp_world_size > 1:
                divisor = device_mesh.tp_world_size * 128
                padded_vocab_size = ((vocab_size + divisor - 1) // divisor) * divisor
            else:
                padded_vocab_size = vocab_size

        num_attention_heads = getattr(text_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(text_config, 'num_key_value_heads', num_attention_heads)
        hidden_size = getattr(text_config, 'hidden_size', 4096)

        # Get kv_channels (head_dim)
        kv_channels = getattr(text_config, 'head_dim', None)
        if kv_channels is None:
            kv_channels = hidden_size // num_attention_heads

        # Get rope_scaling
        rope_scaling = getattr(text_config, 'rope_scaling', None)

        model_type = getattr(hf_config, 'model_type', 'qwen2')

        # Detect multimodal models without importing the Megatron registry.
        # The registry import chain can pull in megatron.core, which must stay
        # behind the MindSpeed bootstrap on NPU.
        from .model.constant import MLLMModelType
        is_multimodal = model_type in {value for key, value in vars(MLLMModelType).items() if not key.startswith('_')}

        # Determine QKV bias
        if hasattr(text_config, 'attention_bias'):
            add_qkv_bias = text_config.attention_bias
        elif model_type in ('qwen2', 'qwen2_5', 'qwen2_vl', 'qwen2_5_vl'):
            add_qkv_bias = True
        else:
            add_qkv_bias = False

        # Determine QK layernorm
        qk_layernorm = (getattr(text_config, 'qk_layernorm', False) or getattr(text_config, 'use_qk_norm', False))
        # MoE config
        num_experts = (
            getattr(text_config, 'num_experts', 0) or getattr(text_config, 'n_routed_experts', 0)
            or getattr(text_config, 'num_local_experts', 0) or 0)
        num_experts_per_tok = (
            getattr(text_config, 'num_experts_per_tok', 2) or getattr(text_config, 'moe_topk', 2) or 2)
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

    def create_model(self, ) -> List[nn.Module]:
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

        from .model.gpt_model import GPTModel
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
        num_query_groups = mg_config_dict.get('num_query_groups', num_attention_heads)
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

        # Custom finalize_model_grads for LoRA, registered via TransformerConfig.
        # Fixes two issues with Megatron's native finalize_model_grads:
        #
        # 1. Bare models (single-rank / no-op wrap) only carry ddp_config but lack
        #    finish_grad_sync(), so we gate on real DDP capability instead.
        #
        # 2. In multi-rank LoRA + PP, native _allreduce_word_embedding_grads assumes
        #    shared embedding/output weight always has a grad. LoRA freezes the base
        #    weight so grad is None -> all_reduce(None) crashes. We monkey-patch that
        #    one helper to skip None grads, reusing the rest of native finalize via
        #    try/finally to avoid forking the entire module.
        from megatron.core.distributed import finalize_model_grads as _native_finalize_model_grads

        def finalize_model_grads_for_lora(model, *args, **kwargs):
            import importlib
            from megatron.core import parallel_state
            from megatron.core.distributed import DistributedDataParallel as MegatronDDP
            from megatron.core.distributed.finalize_model_grads import (_get_main_grad_attr, _reshard_if_dtensor,
                                                                        _unshard_if_dtensor, get_attr_wrapped_model)
            from peft import PeftModel as _PeftModel

            # Unwrap PeftModel -> LoraModel -> real model to check DDP capability.
            def _get_base_model(m):
                if isinstance(m, _PeftModel):
                    return _get_base_model(m.base_model.model)
                return m

            # Fix 1: check real DDP capability, not just ddp_config presence.
            base_model = _get_base_model(model[0])
            if isinstance(base_model, MegatronDDP) or hasattr(base_model, 'finish_grad_sync'):
                # Fix 2: temporarily swap in the None-safe embedding allreduce.
                finalize_model_grads_mod = importlib.import_module('megatron.core.distributed.finalize_model_grads')
                orig_allreduce_word_embedding_grads = finalize_model_grads_mod._allreduce_word_embedding_grads
                finalize_model_grads_mod._allreduce_word_embedding_grads = _allreduce_word_embedding_grads_allow_none
                try:
                    return _native_finalize_model_grads(model, *args, **kwargs)
                finally:
                    finalize_model_grads_mod._allreduce_word_embedding_grads = orig_allreduce_word_embedding_grads

            # Bare model (single-rank / no-op wrap): no DDP sync, skip.
            return

        # MoE configuration
        num_experts = mg_config_dict.get('num_experts', 0) or 0
        moe_ffn_hidden_size = mg_config_dict.get('moe_ffn_hidden_size')
        moe_router_topk = mg_config_dict.get('moe_router_topk', 2) or 2
        moe_shared_expert_intermediate_size = mg_config_dict.get('moe_shared_expert_intermediate_size')

        # Build MoE-related kwargs
        moe_kwargs = {}
        if num_experts > 0:
            moe_kwargs.update({
                'num_moe_experts':
                num_experts,
                'moe_router_topk':
                moe_router_topk,
                'moe_router_load_balancing_type':
                mg_config_dict.get('moe_router_load_balancing_type', 'aux_loss'),
                # MoE performance optimizations
                'moe_token_dispatcher_type':
                mg_config_dict.get('moe_token_dispatcher_type',
                                   'alltoall'),  # 'alltoall' is more efficient than 'allgather'
                'moe_grouped_gemm':
                mg_config_dict.get('moe_grouped_gemm',
                                   True),  # Enable for better performance (requires grouped_gemm package)
                'moe_aux_loss_coeff':
                mg_config_dict.get('moe_aux_loss_coeff', 0.0),  # Auxiliary load balancing loss coefficient
            })

            # FFN hidden size for MoE
            if moe_ffn_hidden_size:
                moe_kwargs['moe_ffn_hidden_size'] = moe_ffn_hidden_size

            # Shared expert configuration
            if moe_shared_expert_intermediate_size:
                moe_kwargs['moe_shared_expert_intermediate_size'] = moe_shared_expert_intermediate_size

            # Router score function (sigmoid for Qwen3, softmax for others)
            if mg_config_dict.get('moe_router_score_function'):
                moe_kwargs['moe_router_score_function'] = mg_config_dict['moe_router_score_function']

            # Expert bias for sigmoid router
            if mg_config_dict.get('moe_router_enable_expert_bias'):
                moe_kwargs['moe_router_enable_expert_bias'] = mg_config_dict['moe_router_enable_expert_bias']

        # Sequence parallel requires TP > 1
        # Auto-enable for MoE with TP > 1 (required by Megatron)
        use_sequence_parallel = self.sequence_parallel and self.tp_size > 1
        if num_experts > 0 and self.tp_size > 1 and not use_sequence_parallel:
            use_sequence_parallel = True
            # Sync the flag back so that callers (e.g. padding logic in
            # megatron.py) see the auto-enabled value.
            self.sequence_parallel = True
            if self.device_mesh is not None:
                self.device_mesh.sequence_parallel = True

        # For MoE models, ffn_hidden_size should be moe_ffn_hidden_size if not specified
        ffn_hidden_size = mg_config_dict.get('ffn_hidden_size')
        if ffn_hidden_size is None:
            ffn_hidden_size = moe_ffn_hidden_size or (4 * mg_config_dict['hidden_size'])

        # For models with non-standard head dimensions (like Qwen3-30B-A3B)
        kv_channels = mg_config_dict.get('kv_channels')

        # Activation function for SwiGLU (required by Megatron when gated_linear_unit=True)
        use_swiglu = mg_config_dict.get('swiglu', True)
        activation_func = torch.nn.functional.silu if use_swiglu else torch.nn.functional.gelu

        # Enable bias_activation_fusion for SwiGLU
        # Note: Only works with TransformerEngine and no bias in linear layers
        has_bias = not mg_config_dict.get('disable_bias_linear', True)
        bias_activation_fusion = use_swiglu and not has_bias
        if 'moe_token_dispatcher_type' not in moe_kwargs:
            moe_kwargs['moe_token_dispatcher_type'] = 'alltoall' if self.variable_seq_lengths else 'allgather'
        is_npu = Platform.device_prefix() == 'npu'
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
            expert_model_parallel_size=self.ep_size,
            virtual_pipeline_model_parallel_size=self.vpp_size,
            sequence_parallel=use_sequence_parallel,
            params_dtype=self.params_dtype,
            fp16=self.params_dtype == torch.float16,
            bf16=self.params_dtype == torch.bfloat16,
            pipeline_dtype=self.params_dtype,  # Required when using pipeline parallelism
            use_cpu_initialization=self.use_cpu_initialization,
            add_qkv_bias=self.add_qkv_bias,
            variable_seq_lengths=self.variable_seq_lengths,
            add_bias_linear=not mg_config_dict.get('disable_bias_linear', True),
            gated_linear_unit=use_swiglu,
            activation_func=activation_func,  # SiLU for SwiGLU, GELU otherwise
            bias_activation_fusion=bias_activation_fusion,  # Fused SwiGLU for performance
            normalization='RMSNorm',
            layernorm_epsilon=mg_config_dict.get('norm_epsilon', 1e-6),
            qk_layernorm=mg_config_dict.get('qk_layernorm', False),
            hidden_dropout=0.0,
            attention_dropout=0.0,
            # Performance optimizations
            # NPU fallback: the current environment does not provide the TBE-backed
            # fused softmax kernel that MindSpeed's NPU path selects by default.
            # Keep the GPU fast path unchanged, but fall back to unfused softmax on NPU
            # so attention can run without a hard dependency on `tbe`.
            masked_softmax_fusion=not is_npu,
            bias_dropout_fusion=True,  # Fused bias + dropout
            apply_rope_fusion=True,  # Fused RoPE application
            attention_softmax_in_fp32=True,  # Numerical stability
            attention_backend=AttnBackend.flash,
            # Activation recomputation for memory efficiency
            recompute_granularity=self.recompute_granularity,
            recompute_modules=self.recompute_modules if self.recompute_granularity == 'selective' else None,
            recompute_method=recompute_method,
            recompute_num_layers=recompute_num_layers,
            # Critical: Set finalize_model_grads_func for DP gradient synchronization
            # Uses custom wrapper that handles both DDP and PEFT/LoRA models
            finalize_model_grads_func=finalize_model_grads_for_lora,
            calculate_per_token_loss=True,
            # MoE configuration
            **moe_kwargs,
        )
        if exists('megatron_core>=0.13'):
            config.expert_tensor_parallel_size = self.etp_size

        if mg_config_dict.get('use_shared_expert_gate'):
            config.moe_use_shared_expert_gate = True
        if mg_config_dict.get('rotary_interleaved'):
            config.rotary_interleaved = True
        partial_rotary_factor = mg_config_dict.get('partial_rotary_factor')
        if partial_rotary_factor is not None:
            config.rotary_percent = partial_rotary_factor
            config.apply_rope_fusion = False
        mrope_section = mg_config_dict.get('mrope_section')
        if mrope_section is not None:
            config.mrope_section = mrope_section
        if mg_config_dict.get('mrope_interleaved'):
            config.mrope_interleaved = True

        self.config = config

        # Delegate model-specific config & layer spec construction to the loader
        loader = model_meta.loader() if model_meta else None
        if loader is not None:
            loader.post_config(config, self, mg_config_dict)
            layer_spec = loader.get_layer_spec(config, self, mg_config_dict)
        else:
            from .model.register import MegatronModelLoader
            default_loader = MegatronModelLoader()
            default_loader.post_config(config, self, mg_config_dict)
            layer_spec = default_loader.get_layer_spec(config, self, mg_config_dict)

        # Create model
        max_seq_length = getattr(hf_config, 'max_position_embeddings', 4096)
        rotary_base = mg_config_dict.get('rotary_base', 10000)
        position_embedding_type = mg_config_dict.get('position_embedding_type', 'rope')
        extra_init_args = {}
        if hasattr(hf_config,
                   'rope_scaling') and hf_config.rope_scaling is not None and 'factor' in hf_config.rope_scaling:
            extra_init_args = {'seq_len_interpolation_factor': hf_config.rope_scaling['factor']}
        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            model = []
            has_vp_stage = inspect.signature(mpu.is_pipeline_first_stage).parameters.get('vp_stage', None) is not None
            for i in range(vpp_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                extra_kwargs = {} if not has_vp_stage else {'ignore_virtual': False, 'vp_stage': i}
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
                    share_embeddings_and_output_weights=getattr(hf_config, 'tie_word_embeddings', False),
                    position_embedding_type=position_embedding_type,
                    rotary_base=rotary_base,
                    **extra_init_args)
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
                share_embeddings_and_output_weights=getattr(hf_config, 'tie_word_embeddings', False),
                position_embedding_type=position_embedding_type,
                rotary_base=rotary_base,
                **extra_init_args,
            )
            model = [model]
        self._model = model
        return model
