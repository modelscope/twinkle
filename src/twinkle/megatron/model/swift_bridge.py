# Copyright (c) twinkle authors. All rights reserved.
"""Bridge module for Megatron-Core weight conversion.

TODO: Remove dependency on swift package. The bridge logic should be
implemented independently in twinkle to avoid external dependencies.

This module provides:
1. TwinkleArgs: A dataclass that mimics megatron.training.get_args() return value
2. MegatronBridgeInitializer: Creates Megatron models with proper initialization
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import torch.distributed as dist

try:
    from safetensors.torch import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# Cache for Swift bridge availability check
_SWIFT_BRIDGE_AVAILABLE = None
_SWIFT_GPT_BRIDGE_CLASS = None


def deep_getattr(obj, attr: str, default=None):
    """Get nested attribute from object using dot notation."""
    try:
        for key in attr.split('.'):
            obj = getattr(obj, key)
        return obj
    except AttributeError:
        return default


def is_last_rank() -> bool:
    """Check if current process is the last rank."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == dist.get_world_size() - 1


class LazyTensor:
    """Lazy tensor wrapper for deferred loading."""
    def __init__(self, tensor=None, loader=None):
        self.tensor = tensor
        self.loader = loader

    def load(self):
        if self.tensor is None:
            return self.loader()
        return self.tensor


class SafetensorLazyLoader:
    """Lazy loader for safetensor files."""
    def __init__(self, hf_model_dir: str, is_peft_format: bool = False):
        self.hf_model_dir = hf_model_dir
        self.is_peft_format = is_peft_format
        self._weight_map = {}
        self._file_handles = {}
        self._load_index()

    def _open_file(self, filename: str):
        if filename not in self._file_handles:
            file_path = os.path.join(self.hf_model_dir, filename)
            self._file_handles[filename] = safe_open(file_path, framework='pt')
        return self._file_handles[filename]

    def _load_index(self):
        import json
        index_path = os.path.join(self.hf_model_dir, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self._weight_map = json.load(f).get('weight_map', {})
        else:
            safetensors_fname = 'adapter_model.safetensors' if self.is_peft_format else 'model.safetensors'
            safetensors_file = os.path.join(self.hf_model_dir, safetensors_fname)
            if os.path.exists(safetensors_file):
                with safe_open(safetensors_file, framework='pt') as f:
                    for key in f.keys():
                        self._weight_map[key] = safetensors_fname

    def get_state_dict(self):
        return {k: LazyTensor(loader=partial(self._load_tensor, key=k)) for k in self._weight_map.keys()}

    def _load_tensor(self, key):
        return self._open_file(self._weight_map[key]).get_tensor(key)

    def close(self):
        self._file_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class TwinkleArgs:
    """Args class that mimics megatron.training.get_args() return value.
    
    TODO: Remove swift dependency. This class is currently designed to be compatible
    with external GPTBridge. Once independent bridge logic is implemented, this
    can be simplified.
    """
    # Model architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 32
    num_layers: int = 32
    ffn_hidden_size: int = 11008
    padded_vocab_size: int = 32000
    
    # Model options
    group_query_attention: bool = False
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = False
    multi_latent_attention: bool = False
    untie_embeddings_and_output_weights: bool = True
    
    # MoE
    num_experts: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_enable_expert_bias: bool = False
    
    # MLA (Multi-Latent Attention) - for DeepSeek models
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 32
    
    # MTP (Multi-Token Prediction)
    mtp_num_layers: int = 0
    
    # Parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    
    distributed_timeout_minutes: int = 300000
    distributed_backend: str = 'nccl'
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1
    
    # Paths and identifiers
    model_dir: str = ''
    hf_model_type: str = 'qwen2'
    
    # Task type
    task_type: str = 'causal_lm'
    
    # Save settings
    max_shard_size: str = '5GB'
    
    # Multimodal
    is_multimodal: bool = False
    
    # Hub settings
    use_hf: bool = False
    hub_token: Optional[str] = None
    
    # Additional Megatron settings
    fp16: bool = False
    bf16: bool = True
    accumulate_allreduce_grads_in_fp32: bool = False
    async_tensor_model_parallel_allreduce: bool = False
    use_distributed_optimizer: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    
    # Softmax type
    softmax_type: str = 'vanilla'
    
    # Extra Megatron arguments
    padding_free: bool = True
    mlp_padding_free: bool = False
    check_model: bool = True
    initialize_embedding: bool = False
    rope_scaling: Optional[Any] = None
    torch_dtype: Optional[Any] = None
    model: Optional[str] = None
    model_type: Optional[str] = None
    load_safetensors: Optional[bool] = None
    save_safetensors: bool = True
    adapters: Optional[Any] = None
    merge_lora: Optional[bool] = None
    
    # Training settings
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: str = 'selective'
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None
    use_cpu_initialization: bool = False
    deterministic_mode: bool = False
    no_masked_softmax_fusion: bool = False
    no_bias_dropout_fusion: Optional[bool] = None
    no_bias_swiglu_fusion: bool = False
    no_rope_fusion: Optional[bool] = None
    
    # LoRA settings
    train_type: Optional[str] = None
    lora_rank: int = 8
    lora_alpha: int = 8
    
    @classmethod
    def from_hf_config(cls, hf_config: Any, tp_size: int = 1, pp_size: int = 1, 
                       ep_size: int = 1, etp_size: Optional[int] = None,
                       model_dir: str = '', padded_vocab_size: Optional[int] = None,
                       use_hf: bool = False, hub_token: Optional[str] = None):
        """Create TwinkleArgs from HuggingFace config.
        
        Args:
            hf_config: HuggingFace model configuration.
            tp_size: Tensor parallel size.
            pp_size: Pipeline parallel size.
            ep_size: Expert parallel size.
            etp_size: Expert tensor parallel size (defaults to tp_size).
            model_dir: Path to model directory.
            padded_vocab_size: Padded vocabulary size (auto-computed if None).
            use_hf: Whether to use HuggingFace Hub (vs ModelScope).
            hub_token: Hub token for authentication.
        """
        import os
        
        vocab_size = getattr(hf_config, 'vocab_size', 32000)
        if padded_vocab_size is None:
            # Pad to multiple of tp_size * 128 for efficiency
            divisor = tp_size * 128
            padded_vocab_size = ((vocab_size + divisor - 1) // divisor) * divisor
        
        num_attention_heads = getattr(hf_config, 'num_attention_heads', 32)
        num_query_groups = getattr(hf_config, 'num_key_value_heads', num_attention_heads)
        model_type = getattr(hf_config, 'model_type', 'qwen2')
        
        # Determine QKV bias - Qwen2 has bias by default but config doesn't expose it
        if hasattr(hf_config, 'attention_bias'):
            add_qkv_bias = hf_config.attention_bias
        elif model_type in ('qwen2', 'qwen2_5'):
            add_qkv_bias = True
        else:
            add_qkv_bias = False
        
        # MoE config
        num_experts = getattr(hf_config, 'num_experts', None) or \
                      getattr(hf_config, 'n_routed_experts', None) or \
                      getattr(hf_config, 'num_local_experts', None)
        
        # QK layernorm (Qwen3)
        qk_layernorm = getattr(hf_config, 'qk_layernorm', False) or \
                       getattr(hf_config, 'use_qk_norm', False)
        
        # MLA settings (DeepSeek)
        q_lora_rank = getattr(hf_config, 'q_lora_rank', None)
        multi_latent_attention = q_lora_rank is not None or \
                                 getattr(hf_config, 'kv_lora_rank', None) is not None
        
        # Get distributed settings from environment
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        return cls(
            hidden_size=getattr(hf_config, 'hidden_size', 4096),
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            num_layers=getattr(hf_config, 'num_hidden_layers', 32),
            ffn_hidden_size=getattr(hf_config, 'intermediate_size', 11008),
            padded_vocab_size=padded_vocab_size,
            group_query_attention=num_query_groups != num_attention_heads,
            add_qkv_bias=add_qkv_bias,
            add_bias_linear=getattr(hf_config, 'mlp_bias', False),
            qk_layernorm=qk_layernorm,
            multi_latent_attention=multi_latent_attention,
            untie_embeddings_and_output_weights=not getattr(hf_config, 'tie_word_embeddings', False),
            num_experts=num_experts,
            moe_shared_expert_intermediate_size=getattr(hf_config, 'shared_expert_intermediate_size', None),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=getattr(hf_config, 'kv_lora_rank', 32),
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size or tp_size,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            model_dir=model_dir,
            hf_model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            adapters=[],  # Initialize as empty list
        )


# =============================================================================
# GPTBridge Adapter
# TODO: Implement independent bridge logic to remove swift dependency.
# =============================================================================
def _import_swift_bridge():
    """Import GPTBridge from external package.
    
    TODO: Implement independent bridge logic in twinkle. The weight conversion
    between HuggingFace and Megatron formats should be self-contained.
    
    Returns:
        GPTBridge class if available, None otherwise.
    """
    global _SWIFT_BRIDGE_AVAILABLE, _SWIFT_GPT_BRIDGE_CLASS
    
    if _SWIFT_BRIDGE_AVAILABLE is not None:
        return _SWIFT_GPT_BRIDGE_CLASS
    
    try:
        from swift.utils import disable_safe_ddp_context_use_barrier
        
        with disable_safe_ddp_context_use_barrier():
            from swift.megatron.model.gpt_bridge import GPTBridge
        
        _SWIFT_BRIDGE_AVAILABLE = True
        _SWIFT_GPT_BRIDGE_CLASS = GPTBridge
        return GPTBridge
    except ImportError as e:
        _SWIFT_BRIDGE_AVAILABLE = False
        _SWIFT_GPT_BRIDGE_CLASS = None
        return None
    except Exception as e:
        import traceback
        print(f"Warning: Failed to import GPTBridge: {e}")
        traceback.print_exc()
        _SWIFT_BRIDGE_AVAILABLE = False
        _SWIFT_GPT_BRIDGE_CLASS = None
        return None


def use_swift_bridge() -> bool:
    """Check if GPTBridge is available."""
    _import_swift_bridge()
    return _SWIFT_BRIDGE_AVAILABLE is True


class SwiftBridgeAdapter:
    """Adapter to use swift's GPTBridge with twinkle's TwinkleArgs.
    
    TODO: Remove swift dependency. Implement independent bridge logic in twinkle.
    
    This class wraps swift's GPTBridge for weight loading/saving between
    HuggingFace and Megatron formats.
    """
    
    def __init__(self, args: TwinkleArgs, hf_model=None, disable_tqdm: bool = False):
        self.args = args
        self.hf_model = hf_model
        self.disable_tqdm = disable_tqdm
        self._swift_bridge = None
        
        self._init_swift_bridge()
    
    def _init_swift_bridge(self):
        """Initialize swift's GPTBridge with our args."""
        GPTBridge = _import_swift_bridge()
        if GPTBridge is None:
            raise ImportError(
                "swift package is required for Megatron weight loading. "
                "Please install: pip install ms-swift"
            )
        
        # Use Megatron's official set_args to set global args
        from megatron.training.global_vars import set_args, get_args
        
        # Check if args already set
        try:
            existing_args = get_args()
            # Args already initialized, we'll use existing
            self._swift_bridge = GPTBridge(disable_tqmd=self.disable_tqdm)
        except AssertionError:
            # Args not initialized, set our args
            set_args(self.args)
            self._swift_bridge = GPTBridge(disable_tqmd=self.disable_tqdm)
    
    def load_weights(self, mg_model, hf_model_dir: str, is_peft_format: bool = False):
        """Load weights from HuggingFace checkpoint into Megatron model."""
        self._swift_bridge.load_weights(mg_model, hf_model_dir, is_peft_format)
    
    def save_weights(self, mg_models, output_dir: str, hf_model_dir: str = None, is_peft_format: bool = False):
        """Save weights in HuggingFace format."""
        self._swift_bridge.save_weights(mg_models, output_dir, is_peft_format)


def create_bridge_adapter(
    hf_config: Any,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    model_dir: str = '',
    padded_vocab_size: Optional[int] = None,
) -> SwiftBridgeAdapter:
    """Create a bridge adapter for weight loading/saving.
    
    TODO: Remove swift dependency. Implement independent bridge logic.
    
    Args:
        hf_config: HuggingFace model config.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        ep_size: Expert parallel size.
        model_dir: Path to model directory.
        padded_vocab_size: Padded vocabulary size.
        
    Returns:
        SwiftBridgeAdapter instance.
    """
    args = TwinkleArgs.from_hf_config(
        hf_config,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        model_dir=model_dir,
        padded_vocab_size=padded_vocab_size,
    )
    
    return SwiftBridgeAdapter(args)


def create_megatron_model_with_swift(
    model_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    params_dtype=None,
    use_cpu_initialization: bool = True,
    attention_backend: str = 'unfused',
    load_weights: bool = True,
):
    """Create Megatron model using swift's initialization flow.
    
    TODO: Remove swift dependency. Implement independent initialization logic.
    
    Args:
        model_path: Path to HuggingFace model or model ID.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        ep_size: Expert parallel size.
        params_dtype: Parameter dtype (default: torch.bfloat16).
        use_cpu_initialization: Initialize on CPU first (for memory efficiency).
        attention_backend: Attention backend ('unfused' for precision, 'flash' for speed).
        load_weights: Whether to load weights.
        
    Returns:
        Tuple of (model, bridge, megatron_model_meta).
    """
    import torch
    from transformers import AutoConfig
    
    if params_dtype is None:
        params_dtype = torch.bfloat16
    
    # Download model if needed
    if not os.path.isdir(model_path):
        try:
            from modelscope import snapshot_download
            model_path = snapshot_download(model_path)
        except ImportError:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path)
    
    # Load HF config
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Import Swift modules with barrier disabled
    from swift.utils import disable_safe_ddp_context_use_barrier
    
    with disable_safe_ddp_context_use_barrier():
        from swift.megatron import (
            MegatronArguments, convert_hf_config, get_megatron_model_meta
        )
    
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    
    # Check if Megatron is already initialized
    try:
        existing_args = get_args()
        megatron_initialized = True
    except AssertionError:
        megatron_initialized = False
    
    # Get model meta first to get extra_args_provider
    megatron_model_meta = get_megatron_model_meta(hf_config.model_type)
    if megatron_model_meta is None:
        raise ValueError(f'Model type {hf_config.model_type} not supported by Swift')
    
    if not megatron_initialized:
        # Convert HF config to Megatron config kwargs
        config_kwargs = convert_hf_config(hf_config)
        
        # Create MegatronArguments
        megatron_args = MegatronArguments(
            model=model_path,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            torch_dtype=params_dtype,
            use_cpu_initialization=use_cpu_initialization,
            attention_backend=attention_backend,
            **config_kwargs,
        )
        
        # Parse to Megatron format
        extra_args = megatron_args.parse_to_megatron()
        
        # Initialize Megatron
        extra_args_provider = megatron_model_meta.extra_args_provider
        initialize_megatron(extra_args_provider=extra_args_provider, args_defaults=extra_args)
    
    # Determine pre_process and post_process based on pipeline stage
    from megatron.core import parallel_state as mpu
    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    
    model = megatron_model_meta.model_provider(pre_process=pre_process, post_process=post_process)
    
    # Load weights if requested
    bridge = None
    if load_weights:
        bridge = megatron_model_meta.bridge_cls()
        bridge.load_weights(model, model_path)
    
    return model, bridge, megatron_model_meta


class MegatronBridgeInitializer:
    """Megatron model initializer using bridge-based initialization flow.
    
    TODO: Remove swift dependency. Implement independent initialization logic.
    
    Example:
        initializer = MegatronBridgeInitializer(
            tp_size=2,
            pp_size=1,
            params_dtype=torch.bfloat16,
        )
        model = initializer.create_model('Qwen/Qwen2.5-7B-Instruct')
    """
    
    def __init__(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        params_dtype=None,
        use_cpu_initialization: bool = True,
        attention_backend: str = 'flash',  # Use flash for training performance
    ):
        """Initialize MegatronBridgeInitializer.
        
        Args:
            tp_size: Tensor parallel size.
            pp_size: Pipeline parallel size.
            ep_size: Expert parallel size.
            params_dtype: Parameter dtype (default: torch.bfloat16).
            use_cpu_initialization: Initialize on CPU first.
            attention_backend: Attention backend.
        """
        import torch
        
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.params_dtype = params_dtype if params_dtype is not None else torch.bfloat16
        self.use_cpu_initialization = use_cpu_initialization
        self.attention_backend = attention_backend
        
        self._model = None
        self._bridge = None
        self._model_meta = None
        self._hf_config = None
        
    def create_model(
        self,
        model_path: str,
        load_weights: bool = True,
    ):
        """Create Megatron model from HuggingFace checkpoint.
        
        Args:
            model_path: Path to HuggingFace model or model ID.
            load_weights: Whether to load weights.
            
        Returns:
            Megatron model.
        """
        from transformers import AutoConfig
        
        # Download model if needed
        if not os.path.isdir(model_path):
            try:
                from modelscope import snapshot_download
                model_path = snapshot_download(model_path)
            except ImportError:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(model_path)
        
        # Store HF config
        self._hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        self._model, self._bridge, self._model_meta = create_megatron_model_with_swift(
            model_path=model_path,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            ep_size=self.ep_size,
            params_dtype=self.params_dtype,
            use_cpu_initialization=self.use_cpu_initialization,
            attention_backend=self.attention_backend,
            load_weights=load_weights,
        )
        
        return self._model
    
    @property
    def hf_config(self):
        """Get the HuggingFace config."""
        return self._hf_config
    
    @property
    def bridge(self):
        """Get the Swift bridge instance."""
        return self._bridge
    
    @property
    def model_meta(self):
        """Get the Megatron model meta."""
        return self._model_meta
    
    def load_weights(self, model, model_path: str):
        """Load weights into an existing model.
        
        Args:
            model: Megatron model.
            model_path: Path to HuggingFace checkpoint.
        """
        if self._bridge is None:
            # Create bridge from model meta
            if self._model_meta is None:
                raise ValueError("Must call create_model first or provide model_meta")
            self._bridge = self._model_meta.bridge_cls()
        
        self._bridge.load_weights(model, model_path)
    
    def save_weights(self, models, output_dir: str, is_peft_format: bool = False):
        """Save weights in HuggingFace format.
        
        Args:
            models: Megatron model(s).
            output_dir: Output directory.
            is_peft_format: Whether to save in PEFT format.
        """
        if self._bridge is None:
            raise ValueError("Must load weights first")
        
        if not isinstance(models, (list, tuple)):
            models = [models]
        
        self._bridge.save_weights(models, output_dir, is_peft_format=is_peft_format)
