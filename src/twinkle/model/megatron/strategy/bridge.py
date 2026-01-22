# Copyright (c) ModelScope Contributors. All rights reserved.
# GPT Bridge for HuggingFace to Megatron-Core weight conversion.
import glob
import inspect
import json
import os
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from twinkle import exists, Platform
from twinkle.hub import HubOperation
from twinkle import torch_util


def _deep_getattr(obj, attr: str, default=None):
    """Get nested attribute from object using dot notation."""
    try:
        for key in attr.split('.'):
            obj = getattr(obj, key)
        return obj
    except AttributeError:
        return default


def _is_last_rank() -> bool:
    """Check if current process is the last rank for writing.

    For DP > 1, we want only DP rank 0 to write to avoid conflicts.
    For PP, we want the last PP stage.
    For TP, all TP ranks participate in gather, but only one writes.
    """
    if not dist.is_initialized():
        return True

    from megatron.core import parallel_state as mpu
    if mpu.is_initialized():
        # Only DP rank 0 writes
        dp_rank = mpu.get_data_parallel_rank()
        if dp_rank != 0:
            return False
        # For PP, only last stage needs to write certain weights
        # (handled separately in export_weights)
        return True

    return dist.get_rank() == dist.get_world_size() - 1


class _LazyTensor:
    """Lazy tensor wrapper for deferred loading."""
    def __init__(self, loader, key: str):
        self._loader = loader
        self._key = key

    def load(self) -> torch.Tensor:
        """Load the tensor."""
        return self._loader.get_tensor(self._key)


class _SafetensorLoader:
    """Lazy loader for safetensor files."""
    def __init__(self, model_dir: str, is_peft_format: bool = False):
        self.model_dir = model_dir
        self.is_peft_format = is_peft_format
        self._handles = {}
        self._index = None
        self._key_to_file = {}
        self._load_index()

    def _load_index(self):
        """Load safetensor index file if exists."""
        # Try adapter format first for PEFT
        if self.is_peft_format:
            adapter_file = os.path.join(self.model_dir,
                                        'adapter_model.safetensors')
            if os.path.exists(adapter_file):
                handle = safe_open(adapter_file, framework='pt', device='cpu')
                for key in handle.keys():
                    self._key_to_file[key] = adapter_file
                self._handles[adapter_file] = handle
                return

        # Standard index file
        index_file = os.path.join(self.model_dir,
                                  'model.safetensors.index.json')
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                self._index = json.load(f)
            for key, filename in self._index['weight_map'].items():
                self._key_to_file[key] = os.path.join(self.model_dir, filename)
        else:
            # Single file model
            single_file = os.path.join(self.model_dir, 'model.safetensors')
            if os.path.exists(single_file):
                handle = safe_open(single_file, framework='pt', device='cpu')
                for key in handle.keys():
                    self._key_to_file[key] = single_file
                self._handles[single_file] = handle
            else:
                # Try to find any safetensor file
                files = glob.glob(os.path.join(self.model_dir,
                                               '*.safetensors'))
                for filepath in files:
                    handle = safe_open(filepath, framework='pt', device='cpu')
                    for key in handle.keys():
                        self._key_to_file[key] = filepath
                    self._handles[filepath] = handle

    def _get_handle(self, filepath: str):
        """Get or create file handle."""
        if filepath not in self._handles:
            self._handles[filepath] = safe_open(filepath,
                                                framework='pt',
                                                device='cpu')
        return self._handles[filepath]

    def get_tensor(self, key: str) -> torch.Tensor:
        """Load a single tensor."""
        filepath = self._key_to_file.get(key)
        if filepath is None:
            raise KeyError(f'Tensor key not found: {key}')
        handle = self._get_handle(filepath)
        return handle.get_tensor(key)

    def get_lazy(self, key: str) -> _LazyTensor:
        """Get a lazy tensor reference."""
        if key not in self._key_to_file:
            raise KeyError(f'Tensor key not found: {key}')
        return _LazyTensor(self, key)

    def get_state_dict(self) -> Dict[str, _LazyTensor]:
        """Get lazy state dict."""
        return {key: _LazyTensor(self, key) for key in self._key_to_file}

    def keys(self) -> List[str]:
        """Get all tensor keys."""
        return list(self._key_to_file.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_file

    def close(self):
        """Close all file handles."""
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _StreamingSafetensorSaver:
    """Streaming saver for safetensor files."""
    def __init__(self,
                 save_dir: str,
                 max_shard_size: str = '5GB',
                 is_peft_format: bool = False):
        self.save_dir = save_dir
        self.is_peft_format = is_peft_format
        os.makedirs(save_dir, exist_ok=True)

        # Parse max shard size
        size_str = max_shard_size.upper()
        if size_str.endswith('GB'):
            self.max_shard_bytes = int(float(size_str[:-2]) * 1024**3)
        elif size_str.endswith('MB'):
            self.max_shard_bytes = int(float(size_str[:-2]) * 1024**2)
        else:
            self.max_shard_bytes = int(size_str)

        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_idx = 1
        self.weight_map = {}

    def add_tensor(self, key: str, tensor: torch.Tensor):
        """Add tensor to the current shard."""
        if tensor is None:
            return

        tensor_size = tensor.numel() * tensor.element_size()

        # Flush if needed
        if self.current_shard_size + tensor_size > self.max_shard_bytes and self.current_shard:
            self._flush_shard()

        self.current_shard[key] = tensor.contiguous()
        self.current_shard_size += tensor_size

    def _flush_shard(self):
        """Flush current shard to disk."""
        if not self.current_shard:
            return

        if self.is_peft_format:
            filename = 'adapter_model.safetensors'
        else:
            filename = f'model-{self.shard_idx:05d}-of-XXXXX.safetensors'

        filepath = os.path.join(self.save_dir, filename)
        save_file(self.current_shard, filepath)

        for key in self.current_shard:
            self.weight_map[key] = filename

        self.current_shard = {}
        self.current_shard_size = 0
        self.shard_idx += 1

    def finalize(self):
        """Finalize and write index."""
        self._flush_shard()

        if self.is_peft_format:
            return  # PEFT format doesn't need index

        # Fix shard filenames
        total_shards = self.shard_idx - 1
        if total_shards == 0:
            return

        for old_name in list(self.weight_map.values()):
            new_name = old_name.replace('XXXXX', f'{total_shards:05d}')
            if old_name != new_name:
                old_path = os.path.join(self.save_dir, old_name)
                new_path = os.path.join(self.save_dir, new_name)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                for key in self.weight_map:
                    if self.weight_map[key] == old_name:
                        self.weight_map[key] = new_name

        if total_shards > 1:
            index = {
                'metadata': {
                    'total_size':
                    sum(t.numel() * t.element_size()
                        for t in self.current_shard.values())
                },
                'weight_map': self.weight_map
            }
            with open(
                    os.path.join(self.save_dir,
                                 'model.safetensors.index.json'), 'w') as f:
                json.dump(index, f, indent=2)


@dataclass
class BridgeConfig:
    """Configuration for GPTBridge."""
    # Parallelism
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1

    # Model architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_layers: int = 32
    vocab_size: int = 32000
    padded_vocab_size: int = 32000
    intermediate_size: int = 11008
    kv_channels: int = None  # head_dim, if None will be computed from hidden_size // num_attention_heads

    # Options
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = False
    tie_word_embeddings: bool = False

    # MoE
    num_experts: int = 0
    num_experts_per_tok: int = 2
    shared_expert_intermediate_size: int = 0

    model_type: str = 'qwen2'
    max_shard_size: str = '5GB'

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        padded_vocab_size: Optional[int] = None,
    ) -> 'BridgeConfig':
        """Create BridgeConfig from HuggingFace config."""
        vocab_size = getattr(hf_config, 'vocab_size', 32000)
        if padded_vocab_size is None:
            padded_vocab_size = vocab_size
            # Pad to multiple of 64 for efficiency
            if padded_vocab_size % 64 != 0:
                padded_vocab_size = ((padded_vocab_size // 64) + 1) * 64

        num_attention_heads = getattr(hf_config, 'num_attention_heads', 32)
        num_key_value_heads = getattr(hf_config, 'num_key_value_heads',
                                      num_attention_heads)

        # MoE config
        num_experts = getattr(hf_config, 'num_experts', 0) or \
                      getattr(hf_config, 'n_routed_experts', 0) or \
                      getattr(hf_config, 'num_local_experts', 0)
        num_experts_per_tok = getattr(hf_config, 'num_experts_per_tok', 2) or \
                              getattr(hf_config, 'moe_topk', 2)
        shared_expert_size = getattr(hf_config,
                                     'shared_expert_intermediate_size', 0)

        # Determine QKV bias setting
        # Qwen2 has attention bias by default (hardcoded in transformers),
        # but config doesn't have 'attention_bias' field
        model_type = getattr(hf_config, 'model_type', 'qwen2')
        if hasattr(hf_config, 'attention_bias'):
            add_qkv_bias = hf_config.attention_bias
        elif model_type in ('qwen2', 'qwen2_5'):
            # Qwen2/Qwen2.5 uses bias=True for Q, K, V projections
            add_qkv_bias = True
        else:
            add_qkv_bias = False

        # Determine QK layernorm setting
        # Qwen3 uses QK layernorm but doesn't have explicit config attribute
        qk_layernorm = getattr(hf_config, 'qk_layernorm', False) or \
                       getattr(hf_config, 'use_qk_norm', False)
        if not qk_layernorm and model_type in ('qwen3', 'qwen3_moe'):
            # Qwen3 (dense and MoE) always uses QK layernorm (q_norm, k_norm weights)
            qk_layernorm = True

        # Determine kv_channels (head_dim) - Qwen3 has explicit head_dim
        kv_channels = getattr(hf_config, 'head_dim', None)

        return cls(
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            etp_size=tp_size,
            hidden_size=getattr(hf_config, 'hidden_size', 4096),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_layers=getattr(hf_config, 'num_hidden_layers', 32),
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
            intermediate_size=getattr(hf_config, 'intermediate_size', 11008),
            add_qkv_bias=add_qkv_bias,
            add_bias_linear=getattr(hf_config, 'mlp_bias', False),
            qk_layernorm=qk_layernorm,
            tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings',
                                        False),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            shared_expert_intermediate_size=shared_expert_size,
            model_type=model_type,
            kv_channels=kv_channels,  # Explicit head_dim for Qwen3
        )


class GPTBridge:
    """Bridge for converting weights between HuggingFace and Megatron-Core formats.

    Supports Qwen2.5 / Qwen3 model families.
    """

    # HuggingFace model structure constants (Qwen2/Qwen3 compatible)
    HF_LAYERS_PREFIX = 'model.layers'
    HF_EMBED_KEY = 'model.embed_tokens.weight'
    HF_FINAL_LAYERNORM_KEY = 'model.norm.weight'
    HF_LM_HEAD_KEY = 'lm_head.weight'

    def __init__(self,
                 config: BridgeConfig,
                 hf_config: Any = None,
                 disable_tqdm: bool = False):
        """Initialize the bridge.

        Args:
            config: Bridge configuration.
            hf_config: HuggingFace model config (for reference).
            disable_tqdm: Whether to disable progress bar.
        """
        self.config = config
        self.hf_config = hf_config
        self.disable_tqdm = disable_tqdm or not _is_last_rank()

        # Parallel state
        self.tp_size = config.tp_size
        self.pp_size = config.pp_size
        self.ep_size = config.ep_size
        self.etp_size = config.etp_size

        from megatron.core import parallel_state as mpu
        # Get parallel ranks
        if mpu.is_initialized():
            self.tp_rank = mpu.get_tensor_model_parallel_rank()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            self.tp_group = mpu.get_tensor_model_parallel_group()
            self.pp_group = mpu.get_pipeline_model_parallel_group()
            try:
                self.ep_rank = mpu.get_expert_model_parallel_rank()
                self.ep_group = mpu.get_expert_model_parallel_group()
                self.etp_rank = mpu.get_expert_tensor_parallel_rank()
                self.etp_group = mpu.get_expert_tensor_parallel_group()
            except (AttributeError, AssertionError):
                self.ep_rank = 0
                self.ep_group = None
                self.etp_rank = 0
                self.etp_group = None
        else:
            self.tp_rank = 0
            self.pp_rank = 0
            self.tp_group = None
            self.pp_group = None
            self.ep_rank = 0
            self.ep_group = None
            self.etp_rank = 0
            self.etp_group = None

        # PEFT tracking
        self._is_peft_format = False
        self._adapter_name = 'default'
        self._peft_target_modules: Set[str] = set()
        self._peft_modules_to_save: Set[str] = set()
        self._target_device = None
        self._only_last_rank = False

    def _get_tp_split_dim(self, mg_key: Optional[str]) -> Optional[int]:
        """Determine which dimension to split for tensor parallelism."""
        if mg_key is None:
            return None

        # ColumnParallel (split output dim)
        dim0_keys = {
            'word_embeddings',
            'linear_qkv',
            'output_layer',
            'linear_q_proj',
            'linear_q_up_proj',
            'linear_kv_up_proj',
            'eh_proj',  # MTP
        }
        # RowParallel (split input dim)
        dim1_keys = {'linear_proj', 'linear_fc2'}

        # Handle LoRA keys
        if 'lora_A' not in mg_key and 'lora_B' not in mg_key:
            key_parts = mg_key.rsplit('.', 2)
            if len(key_parts) >= 2:
                key = key_parts[-2]
                suffix = key_parts[-1]

                if suffix == 'layer_norm_weight':
                    return None
                elif key in dim0_keys:
                    return 0
                elif key in {'linear_fc1'} and suffix != 'bias':
                    return 1
                elif key in dim1_keys and suffix != 'bias':
                    return 1
        else:
            # LoRA weights
            key_parts = mg_key.rsplit('.', 3)
            if len(key_parts) >= 2:
                key = key_parts[0]
                lora_name = key_parts[1] if len(key_parts) > 1 else ''
                if lora_name == 'lora_A':
                    if key in dim1_keys:
                        return 1
                elif lora_name == 'lora_B':
                    if key in dim0_keys:
                        return 0
                    elif key == 'linear_fc1':
                        return 1

        return None

    def _split_tp(self,
                  tensor: torch.Tensor,
                  tp_dim: Optional[int],
                  is_expert: bool = False) -> torch.Tensor:
        """Split tensor for tensor parallelism."""
        tp_size = self.etp_size if is_expert else self.tp_size
        tp_rank = self.etp_rank if is_expert else self.tp_rank

        if tp_dim is None or tp_size <= 1:
            return tensor
        return tensor.chunk(tp_size, dim=tp_dim)[tp_rank]

    def _all_gather_tp(self,
                       tensor: Optional[torch.Tensor],
                       tp_dim: Optional[int],
                       is_expert: bool = False) -> Optional[torch.Tensor]:
        """All-gather tensor across tensor parallel group."""
        if tensor is None:
            return None

        tensor = tensor.to('cuda')
        tp_size = self.etp_size if is_expert else self.tp_size
        tp_group = self.etp_group if is_expert else self.tp_group

        if tp_dim is None or tp_size <= 1:
            return tensor

        if tp_dim == 0:
            tensor_shape = list(tensor.shape)
            tensor_shape[0] *= tp_size
            output = tensor.new_empty(tensor_shape)
            dist.all_gather_into_tensor(output, tensor, group=tp_group)
            return output
        else:
            output = [torch.empty_like(tensor) for _ in range(tp_size)]
            dist.all_gather(output, tensor, group=tp_group)
            return torch.cat(output, dim=tp_dim)

    def _set_weight(
        self,
        mg_param: Union[torch.Tensor, nn.Parameter, List],
        hf_weight: torch.Tensor,
        mg_key: str,
        is_expert: bool = False,
    ):
        """Set weight from HuggingFace to Megatron parameter."""
        tp_dim = self._get_tp_split_dim(mg_key)
        tensor = self._split_tp(hf_weight, tp_dim, is_expert)

        if not isinstance(mg_param, (list, tuple)):
            mg_param = [mg_param]

        tensor_list = tensor.chunk(len(mg_param), dim=0)
        for i, param in enumerate(mg_param):
            t = tensor_list[i].reshape(*param.shape)
            param.data.copy_(t)

    def _get_weight(
        self,
        mg_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        mg_key: Optional[str],
        is_expert: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get weight from Megatron parameter, gathered across TP."""
        if mg_weight is None:
            return None, None

        tensor = mg_weight
        if not isinstance(tensor, (list, tuple)):
            tensor = [tensor]

        tensor = torch.cat(tensor, dim=0)
        tp_dim = self._get_tp_split_dim(mg_key)
        tensor = self._all_gather_tp(tensor, tp_dim, is_expert)

        if self._target_device is not None and tensor is not None:
            tensor = tensor.to(device=self._target_device)

        if self._only_last_rank and not _is_last_rank():
            return None, None

        return tensor, None

    # =========================================================================
    # Weight Loading Methods
    # =========================================================================

    def _load_embedding(self, mg_model, loader: _SafetensorLoader):
        """Load embedding weights."""
        embed_module = _deep_getattr(mg_model, 'embedding.word_embeddings')
        if embed_module is None:
            return

        hf_weight = loader.get_tensor(self.HF_EMBED_KEY)

        # Pad vocabulary if needed
        if hf_weight.shape[0] < self.config.padded_vocab_size:
            hf_weight = F.pad(
                hf_weight,
                (0, 0, 0, self.config.padded_vocab_size - hf_weight.shape[0]))

        self._set_weight(embed_module.weight, hf_weight,
                         'word_embeddings.weight')

    def _load_output_layer(self, mg_model, loader: _SafetensorLoader):
        """Load output layer (lm_head) weights."""
        output_module = _deep_getattr(mg_model, 'output_layer')
        if output_module is None or output_module.weight is None:
            return

        # Check if weights are tied
        if self.config.tie_word_embeddings:
            hf_weight = loader.get_tensor(self.HF_EMBED_KEY)
        else:
            hf_weight = loader.get_tensor(self.HF_LM_HEAD_KEY)

        # Pad vocabulary if needed
        if hf_weight.shape[0] < self.config.padded_vocab_size:
            hf_weight = F.pad(
                hf_weight,
                (0, 0, 0, self.config.padded_vocab_size - hf_weight.shape[0]))

        self._set_weight(output_module.weight, hf_weight,
                         'output_layer.weight')

    def _load_final_layernorm(self, mg_model, loader: _SafetensorLoader):
        """Load final layer norm weights."""
        ln_module = _deep_getattr(mg_model, 'decoder.final_layernorm')
        if ln_module is None:
            return

        hf_weight = loader.get_tensor(self.HF_FINAL_LAYERNORM_KEY)
        ln_module.weight.data.copy_(hf_weight)

    def _load_attention(self, mg_layer, loader: _SafetensorLoader,
                        layer_idx: int):
        """Load attention layer weights."""
        mg_attn = mg_layer.self_attention
        prefix = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.self_attn.'

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        # Use kv_channels (head_dim) from config if available (for Qwen3 etc.)
        head_dim = getattr(self.config, 'kv_channels',
                           hidden_size // num_heads)
        heads_per_group = num_heads // num_kv_heads

        # Load Q, K, V weights and merge into linear_qkv
        q_weight = loader.get_tensor(f'{prefix}q_proj.weight')
        k_weight = loader.get_tensor(f'{prefix}k_proj.weight')
        v_weight = loader.get_tensor(f'{prefix}v_proj.weight')

        # Infer head_dim from actual weight shapes if needed
        actual_kv_dim = k_weight.shape[0] // num_kv_heads
        if actual_kv_dim != head_dim:
            head_dim = actual_kv_dim

        # Reshape for GQA
        q_weight = q_weight.reshape(num_kv_heads, heads_per_group * head_dim,
                                    hidden_size)
        k_weight = k_weight.reshape(num_kv_heads, head_dim, hidden_size)
        v_weight = v_weight.reshape(num_kv_heads, head_dim, hidden_size)

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1)
        qkv_weight = qkv_weight.reshape(-1, hidden_size)

        self._set_weight(mg_attn.linear_qkv.weight, qkv_weight,
                         'linear_qkv.weight')

        # Load O projection
        o_weight = loader.get_tensor(f'{prefix}o_proj.weight')
        self._set_weight(mg_attn.linear_proj.weight, o_weight,
                         'linear_proj.weight')

        # Load biases if present
        if self.config.add_qkv_bias:
            try:
                q_bias = loader.get_tensor(f'{prefix}q_proj.bias')
                k_bias = loader.get_tensor(f'{prefix}k_proj.bias')
                v_bias = loader.get_tensor(f'{prefix}v_proj.bias')

                # Infer head_dim from actual bias shapes if needed
                actual_bias_head_dim = k_bias.shape[0] // num_kv_heads

                q_bias = q_bias.reshape(num_kv_heads,
                                        heads_per_group * actual_bias_head_dim)
                k_bias = k_bias.reshape(num_kv_heads, actual_bias_head_dim)
                v_bias = v_bias.reshape(num_kv_heads, actual_bias_head_dim)

                qkv_bias = torch.cat([q_bias, k_bias, v_bias],
                                     dim=1).reshape(-1)
                self._set_weight(mg_attn.linear_qkv.bias, qkv_bias,
                                 'linear_qkv.bias')
            except KeyError:
                pass

        # Load input layernorm (may be fused)
        ln_key = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.input_layernorm.weight'
        ln_weight = loader.get_tensor(ln_key)

        ln_param = _deep_getattr(mg_attn, 'linear_qkv.layer_norm_weight')
        if ln_param is not None:
            ln_param.data.copy_(ln_weight)
        else:
            ln_module = _deep_getattr(mg_layer, 'input_layernorm')
            if ln_module is not None:
                ln_module.weight.data.copy_(ln_weight)

        # QK layernorm (Qwen3)
        if self.config.qk_layernorm:
            try:
                q_norm = loader.get_tensor(f'{prefix}q_norm.weight')
                k_norm = loader.get_tensor(f'{prefix}k_norm.weight')
                q_ln = _deep_getattr(mg_attn, 'q_layernorm')
                k_ln = _deep_getattr(mg_attn, 'k_layernorm')
                if q_ln is not None:
                    q_ln.weight.data.copy_(q_norm)
                if k_ln is not None:
                    k_ln.weight.data.copy_(k_norm)
            except KeyError:
                pass

    def _load_mlp(self, mg_layer, loader: _SafetensorLoader, layer_idx: int):
        """Load MLP layer weights."""
        mg_mlp = mg_layer.mlp
        prefix = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.mlp.'

        # Check if gate_up_proj is fused
        try:
            gate_weight = loader.get_tensor(f'{prefix}gate_proj.weight')
            up_weight = loader.get_tensor(f'{prefix}up_proj.weight')

            # Stack gate and up projections (shape: [2, intermediate, hidden])
            fc1_weight = torch.stack([gate_weight, up_weight], dim=0)
            self._set_weight(mg_mlp.linear_fc1.weight, fc1_weight,
                             'linear_fc1.weight')
        except KeyError:
            # Try gate_up_proj (fused)
            try:
                gate_up_weight = loader.get_tensor(
                    f'{prefix}gate_up_proj.weight')
                gate_up_weight = gate_up_weight.view(2, -1,
                                                     gate_up_weight.shape[-1])
                self._set_weight(mg_mlp.linear_fc1.weight, gate_up_weight,
                                 'linear_fc1.weight')
            except KeyError:
                pass

        # Load down projection
        try:
            down_weight = loader.get_tensor(f'{prefix}down_proj.weight')
            self._set_weight(mg_mlp.linear_fc2.weight, down_weight,
                             'linear_fc2.weight')
        except KeyError:
            pass

        # Load post attention layernorm
        ln_key = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.post_attention_layernorm.weight'
        try:
            ln_weight = loader.get_tensor(ln_key)

            ln_param = _deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
            if ln_param is not None:
                ln_param.data.copy_(ln_weight)
            else:
                ln_module = _deep_getattr(mg_layer, 'pre_mlp_layernorm')
                if ln_module is not None:
                    ln_module.weight.data.copy_(ln_weight)
        except KeyError:
            pass

    def _load_moe(self, mg_layer, loader: _SafetensorLoader, layer_idx: int):
        """Load MoE layer weights.

        Handles Expert Parallel (EP) sharding - each EP rank loads only its
        assigned subset of experts based on ep_rank and ep_size.

        For EP=2 with 128 experts:
          - EP rank 0 loads experts 0-63
          - EP rank 1 loads experts 64-127
        """
        mg_mlp = mg_layer.mlp
        prefix = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.mlp.'

        # Load router (replicated across all ranks)
        try:
            router_key = None
            for key in ['gate.weight', 'router.weight', 'gate.wg.weight']:
                full_key = f'{prefix}{key}'
                if full_key in loader:
                    router_key = full_key
                    break

            if router_key:
                router_weight = loader.get_tensor(router_key)
                router_module = _deep_getattr(mg_mlp, 'router')
                if router_module is not None and hasattr(
                        router_module, 'weight'):
                    router_module.weight.data.copy_(router_weight)

            # Load expert bias if present (for sigmoid routers like Qwen3)
            for bias_key in [
                    'gate.e_score_correction_bias',
                    'moe_statics.e_score_correction_bias'
            ]:
                full_bias_key = f'{prefix}{bias_key}'
                if full_bias_key in loader:
                    try:
                        expert_bias = loader.get_tensor(full_bias_key)
                        if router_module is not None and hasattr(
                                router_module, 'expert_bias'):
                            router_module.expert_bias.data.copy_(expert_bias)
                        break
                    except KeyError:
                        continue
        except KeyError:
            pass

        # Load shared experts if present
        if self.config.shared_expert_intermediate_size > 0:
            for shared_key in [
                    'shared_expert', 'shared_experts', 'shared_mlp'
            ]:
                try:
                    gate_weight = loader.get_tensor(
                        f'{prefix}{shared_key}.gate_proj.weight')
                    up_weight = loader.get_tensor(
                        f'{prefix}{shared_key}.up_proj.weight')
                    down_weight = loader.get_tensor(
                        f'{prefix}{shared_key}.down_proj.weight')

                    shared_module = _deep_getattr(mg_mlp, 'shared_experts')
                    if shared_module is not None:
                        fc1_weight = torch.stack([gate_weight, up_weight],
                                                 dim=0)
                        self._set_weight(shared_module.linear_fc1.weight,
                                         fc1_weight, 'linear_fc1.weight')
                        self._set_weight(shared_module.linear_fc2.weight,
                                         down_weight, 'linear_fc2.weight')
                    break
                except KeyError:
                    continue

            # Load shared expert gate if present
            for gate_key in ['shared_expert_gate.weight']:
                full_gate_key = f'{prefix}{gate_key}'
                if full_gate_key in loader:
                    try:
                        gate_weight = loader.get_tensor(full_gate_key)
                        shared_module = _deep_getattr(mg_mlp, 'shared_experts')
                        if shared_module is not None and hasattr(
                                shared_module, 'gate_weight'):
                            shared_module.gate_weight.data.copy_(gate_weight)
                        break
                    except KeyError:
                        continue

        # Load experts with EP sharding
        num_local_experts = self.config.num_experts // self.ep_size
        start_expert_idx = self.ep_rank * num_local_experts
        experts_module = _deep_getattr(mg_mlp, 'experts')

        if experts_module is not None:
            # Determine expert module type
            if hasattr(experts_module, 'weight1'):
                # GroupedMLP format - weights are merged: [hidden, num_experts * ffn_hidden]
                # Need to collect all experts and set at once
                fc1_weights = []  # gate and up weights interleaved
                fc2_weights = []  # down weights

                for local_idx in range(num_local_experts):
                    global_idx = start_expert_idx + local_idx
                    try:
                        gate_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.gate_proj.weight')
                        up_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.up_proj.weight')
                        down_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.down_proj.weight')

                        # Stack gate and up for gated linear unit
                        fc1_weights.append(gate_weight)  # [ffn_hidden, hidden]
                        fc1_weights.append(up_weight)  # [ffn_hidden, hidden]
                        fc2_weights.append(down_weight)  # [hidden, ffn_hidden]
                    except KeyError as e:
                        print(
                            f'Warning: Missing expert {global_idx} weights: {e}'
                        )
                        continue

                if fc1_weights and fc2_weights:
                    # GroupedMLP weight1: [hidden, num_experts * 2 * ffn_hidden] (transposed)
                    # HF format: [num_experts * 2, ffn_hidden, hidden]
                    fc1_stacked = torch.cat(
                        fc1_weights,
                        dim=0)  # [num_experts*2*ffn_hidden, hidden]
                    fc1_stacked = fc1_stacked.t().contiguous(
                    )  # [hidden, num_experts*2*ffn_hidden]

                    # GroupedMLP weight2: [num_experts * ffn_hidden, hidden]
                    fc2_stacked = torch.cat(
                        fc2_weights, dim=0)  # [num_experts*hidden, ffn_hidden]

                    # Set weights directly
                    if experts_module.weight1.shape == fc1_stacked.shape:
                        experts_module.weight1.data.copy_(fc1_stacked)
                    else:
                        # Handle TP split
                        tp_rank = self.etp_rank
                        tp_size = self.etp_size
                        if tp_size > 1:
                            # Split along last dim for weight1
                            chunk_size = fc1_stacked.shape[1] // tp_size
                            fc1_chunk = fc1_stacked[:, tp_rank *
                                                    chunk_size:(tp_rank + 1) *
                                                    chunk_size]
                            experts_module.weight1.data.copy_(fc1_chunk)
                        else:
                            experts_module.weight1.data.copy_(fc1_stacked)

                    if experts_module.weight2.shape == fc2_stacked.shape:
                        experts_module.weight2.data.copy_(fc2_stacked)
                    else:
                        # Handle TP split
                        tp_rank = self.etp_rank
                        tp_size = self.etp_size
                        if tp_size > 1:
                            # Split along first dim for weight2
                            chunk_size = fc2_stacked.shape[0] // tp_size
                            fc2_chunk = fc2_stacked[tp_rank *
                                                    chunk_size:(tp_rank + 1) *
                                                    chunk_size, :]
                            experts_module.weight2.data.copy_(fc2_chunk)
                        else:
                            experts_module.weight2.data.copy_(fc2_stacked)

            elif hasattr(experts_module, 'local_experts'):
                # SequentialMLP format with local_experts list
                for local_idx in range(num_local_experts):
                    global_idx = start_expert_idx + local_idx
                    try:
                        gate_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.gate_proj.weight')
                        up_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.up_proj.weight')
                        down_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.down_proj.weight')

                        expert = experts_module.local_experts[local_idx]
                        if hasattr(expert, 'linear_fc1'):
                            fc1_weight = torch.stack([gate_weight, up_weight],
                                                     dim=0)
                            self._set_weight(expert.linear_fc1.weight,
                                             fc1_weight, 'linear_fc1.weight')
                            self._set_weight(expert.linear_fc2.weight,
                                             down_weight, 'linear_fc2.weight')
                    except KeyError:
                        continue

            elif hasattr(experts_module, 'linear_fc1'):
                # TEGroupedLinear format - weights stored as weight0, weight1, etc.
                for local_idx in range(num_local_experts):
                    global_idx = start_expert_idx + local_idx
                    try:
                        gate_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.gate_proj.weight')
                        up_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.up_proj.weight')
                        down_weight = loader.get_tensor(
                            f'{prefix}experts.{global_idx}.down_proj.weight')

                        fc1_weight = torch.stack([gate_weight, up_weight],
                                                 dim=0)
                        fc1_param = getattr(experts_module.linear_fc1,
                                            f'weight{local_idx}', None)
                        if fc1_param is not None:
                            self._set_weight(fc1_param,
                                             fc1_weight,
                                             'linear_fc1.weight',
                                             is_expert=True)

                        fc2_param = getattr(experts_module.linear_fc2,
                                            f'weight{local_idx}', None)
                        if fc2_param is not None:
                            self._set_weight(fc2_param,
                                             down_weight,
                                             'linear_fc2.weight',
                                             is_expert=True)
                    except KeyError:
                        continue

        # Load post attention layernorm (pre_mlp_layernorm for MoE)
        ln_key = f'{self.HF_LAYERS_PREFIX}.{layer_idx}.post_attention_layernorm.weight'
        try:
            ln_weight = loader.get_tensor(ln_key)
            # Try pre_mlp_layernorm first (used in MoE layers)
            ln_module = _deep_getattr(mg_layer, 'pre_mlp_layernorm')
            if ln_module is not None and hasattr(ln_module, 'weight'):
                ln_module.weight.data.copy_(ln_weight)
            else:
                # Fallback to linear_fc1.layer_norm_weight
                ln_param = _deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
                if ln_param is not None:
                    ln_param.data.copy_(ln_weight)
        except KeyError:
            pass

    def _load_layer(self, mg_layer, loader: _SafetensorLoader, layer_idx: int):
        """Load a single transformer layer."""
        self._load_attention(mg_layer, loader, layer_idx)

        # Check if MoE layer
        if self.config.num_experts > 0:
            self._load_moe(mg_layer, loader, layer_idx)
        else:
            self._load_mlp(mg_layer, loader, layer_idx)

    def load_weights(
        self,
        mg_model: nn.Module,
        model_path: str,
        is_peft_format: bool = False,
        adapter_name: str = 'default',
    ) -> None:
        """Load HuggingFace weights into Megatron model.

        Args:
            mg_model: Megatron GPT model.
            model_path: Path to HuggingFace checkpoint.
            is_peft_format: Whether loading PEFT adapter weights.
            adapter_name: Name of the adapter for PEFT.
        """
        self._is_peft_format = is_peft_format
        self._adapter_name = adapter_name

        with torch.no_grad():
            with _SafetensorLoader(model_path,
                                  is_peft_format=is_peft_format) as loader:
                if is_peft_format:
                    self._load_peft_weights(mg_model, loader)
                else:
                    self._load_base_weights(mg_model, loader)

    def _load_base_weights(self, mg_model: nn.Module,
                           loader: _SafetensorLoader):
        """Load base model weights."""
        # Get decoder
        decoder = _deep_getattr(mg_model, 'decoder')
        if decoder is None:
            decoder = mg_model

        layers = getattr(decoder, 'layers', [])

        # Load pre-process (embedding) on first PP rank
        if self.pp_size <= 1 or self.pp_rank == 0:
            try:
                self._load_embedding(mg_model, loader)
            except Exception as e:
                print(f'Warning: Failed to load embedding: {e}')

        # Load transformer layers
        prog_bar = tqdm(layers,
                        desc='Loading weights',
                        disable=self.disable_tqdm)
        for mg_layer in prog_bar:
            layer_idx = mg_layer.layer_number - 1  # 1-indexed to 0-indexed
            try:
                self._load_layer(mg_layer, loader, layer_idx)
            except Exception as e:
                print(f'Warning: Failed to load layer {layer_idx}: {e}')

        # Load post-process on last PP rank
        if self.pp_size <= 1 or self.pp_rank == self.pp_size - 1:
            try:
                self._load_final_layernorm(mg_model, loader)
                self._load_output_layer(mg_model, loader)
            except Exception as e:
                print(f'Warning: Failed to load post-process: {e}')

    def _load_peft_weights(self, mg_model: nn.Module,
                           loader: _SafetensorLoader):
        """Load PEFT/LoRA adapter weights."""
        state_dict = loader.get_state_dict()
        hf_prefix = 'base_model.model.' if self._is_peft_format else ''

        # Build mapping from HF keys to Megatron keys
        for key, lazy_tensor in state_dict.items():
            # Remove base_model.model. prefix
            if key.startswith(hf_prefix):
                key = key[len(hf_prefix):]

            # Parse the key to find target module
            if '.lora_A.' in key or '.lora_B.' in key:
                tensor = lazy_tensor.load()
                self._load_peft_tensor(mg_model, key, tensor)

    def _load_peft_tensor(self, mg_model: nn.Module, key: str,
                          tensor: torch.Tensor):
        """Load a single PEFT tensor into the model."""
        # Parse key: model.layers.0.self_attn.q_proj.lora_A.weight
        parts = key.split('.')

        # Find layer index
        layer_idx = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
                break

        if layer_idx is None:
            return

        # Get layer
        decoder = _deep_getattr(mg_model, 'decoder')
        if decoder is None:
            decoder = mg_model

        layers = getattr(decoder, 'layers', [])
        for layer in layers:
            if layer.layer_number - 1 == layer_idx:
                mg_layer = layer
                break
        else:
            return

        # Determine target and lora type
        is_lora_A = '.lora_A.' in key
        is_lora_B = '.lora_B.' in key

        if 'self_attn' in key:
            mg_attn = mg_layer.self_attention
            if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key:
                target = _deep_getattr(mg_attn, 'linear_qkv')
            elif 'o_proj' in key:
                target = _deep_getattr(mg_attn, 'linear_proj')
            else:
                return
        elif 'mlp' in key:
            mg_mlp = mg_layer.mlp
            if 'gate_proj' in key or 'up_proj' in key:
                target = _deep_getattr(mg_mlp, 'linear_fc1')
            elif 'down_proj' in key:
                target = _deep_getattr(mg_mlp, 'linear_fc2')
            else:
                return
        else:
            return

        if target is None:
            return

        # Get LoRA module
        if is_lora_A:
            lora_module = _deep_getattr(target, f'lora_A.{self._adapter_name}')
        else:
            lora_module = _deep_getattr(target, f'lora_B.{self._adapter_name}')

        if lora_module is not None and hasattr(lora_module, 'weight'):
            lora_module.weight.data.copy_(tensor)

    # =========================================================================
    # Weight Saving Methods
    # =========================================================================

    def export_weights(
        self,
        mg_models: Union[nn.Module, List[nn.Module]],
        target_device: Optional[str] = None,
        only_last_rank: bool = False,
        is_peft_format: bool = False,
        tqdm_desc: str = 'Exporting: ',
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Export weights from Megatron model to HuggingFace format.

        Yields:
            Tuples of (key, tensor) for each weight.
        """
        self._target_device = target_device
        self._only_last_rank = only_last_rank
        self._is_peft_format = is_peft_format
        self._adapter_name = 'default'
        self._peft_target_modules = set()
        self._peft_modules_to_save = set()

        if not isinstance(mg_models, (list, tuple)):
            mg_models = [mg_models]

        hf_prefix = 'base_model.model.' if is_peft_format else ''

        with torch.no_grad():
            # For now, handle single model
            mg_model = mg_models[0]

            decoder = _deep_getattr(mg_model, 'decoder')
            if decoder is None:
                decoder = mg_model

            layers = getattr(decoder, 'layers', [])

            if not is_peft_format:
                # Export embedding
                if self.pp_size <= 1 or self.pp_rank == 0:
                    embed = _deep_getattr(mg_model,
                                         'embedding.word_embeddings.weight')
                    if embed is not None:
                        weight, _ = self._get_weight(embed.data,
                                                     'word_embeddings.weight')
                        if weight is not None:
                            weight = weight[:self.config.vocab_size]
                            yield f'{hf_prefix}{self.HF_EMBED_KEY}', weight

            # Export layers
            prog_bar = tqdm(layers, desc=tqdm_desc, disable=self.disable_tqdm)
            for mg_layer in prog_bar:
                layer_idx = mg_layer.layer_number - 1
                yield from self._export_layer(mg_layer, layer_idx, hf_prefix,
                                              is_peft_format)

            if not is_peft_format:
                # Export final layernorm and output layer
                if self.pp_size <= 1 or self.pp_rank == self.pp_size - 1:
                    ln_module = _deep_getattr(mg_model,
                                             'decoder.final_layernorm')
                    if ln_module is not None:
                        yield f'{hf_prefix}{self.HF_FINAL_LAYERNORM_KEY}', ln_module.weight.data.clone(
                        )

                    output = _deep_getattr(mg_model, 'output_layer.weight')
                    if output is not None:
                        weight, _ = self._get_weight(output.data,
                                                     'output_layer.weight')
                        if weight is not None:
                            weight = weight[:self.config.vocab_size]
                            yield f'{hf_prefix}{self.HF_LM_HEAD_KEY}', weight

    def _export_layer(
        self,
        mg_layer,
        layer_idx: int,
        hf_prefix: str,
        is_peft_format: bool,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Export a single layer."""
        prefix = f'{hf_prefix}{self.HF_LAYERS_PREFIX}.{layer_idx}.'

        mg_attn = mg_layer.self_attention
        mg_mlp = mg_layer.mlp

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        head_dim = hidden_size // num_heads
        heads_per_group = num_heads // num_kv_heads
        q_dim = heads_per_group * head_dim
        kv_dim = head_dim

        if not is_peft_format:
            # Export QKV
            qkv_weight, _ = self._get_weight(mg_attn.linear_qkv.weight.data,
                                             'linear_qkv.weight')
            if qkv_weight is not None:
                qkv_weight = qkv_weight.reshape(num_kv_heads, -1, hidden_size)
                yield f'{prefix}self_attn.q_proj.weight', qkv_weight[:, :
                                                                     q_dim, :].reshape(
                                                                         -1,
                                                                         hidden_size
                                                                     ).clone()
                yield f'{prefix}self_attn.k_proj.weight', qkv_weight[:, q_dim:
                                                                     q_dim +
                                                                     kv_dim, :].reshape(
                                                                         -1,
                                                                         hidden_size
                                                                     ).clone()
                yield f'{prefix}self_attn.v_proj.weight', qkv_weight[:,
                                                                     -kv_dim:, :].reshape(
                                                                         -1,
                                                                         hidden_size
                                                                     ).clone()

            # Export O
            o_weight, _ = self._get_weight(mg_attn.linear_proj.weight.data,
                                           'linear_proj.weight')
            if o_weight is not None:
                yield f'{prefix}self_attn.o_proj.weight', o_weight

            # Export layernorms
            ln = _deep_getattr(mg_attn, 'linear_qkv.layer_norm_weight')
            if ln is not None:
                yield f'{prefix}input_layernorm.weight', ln.data.clone()

            # Export MLP
            fc1_weight, _ = self._get_weight(mg_mlp.linear_fc1.weight.data,
                                             'linear_fc1.weight')
            if fc1_weight is not None:
                fc1_weight = fc1_weight.view(2, -1, hidden_size)
                yield f'{prefix}mlp.gate_proj.weight', fc1_weight[0].clone()
                yield f'{prefix}mlp.up_proj.weight', fc1_weight[1].clone()

            fc2_weight, _ = self._get_weight(mg_mlp.linear_fc2.weight.data,
                                             'linear_fc2.weight')
            if fc2_weight is not None:
                yield f'{prefix}mlp.down_proj.weight', fc2_weight

            ln2 = _deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
            if ln2 is not None:
                yield f'{prefix}post_attention_layernorm.weight', ln2.data.clone(
                )
        else:
            # Export LoRA weights only
            yield from self._export_lora_layer(mg_attn, mg_mlp, prefix)

    def _export_lora_layer(
        self,
        mg_attn,
        mg_mlp,
        prefix: str,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Export LoRA weights from a layer."""
        # Check if LoRA is applied
        from ..tuners import LoraParallelLinear

        # Attention LoRA
        if isinstance(mg_attn.linear_qkv, LoraParallelLinear):
            lora_A = _deep_getattr(mg_attn.linear_qkv,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = _deep_getattr(mg_attn.linear_qkv,
                                  f'lora_B.{self._adapter_name}.weight')

            if lora_A is not None and lora_B is not None:
                lora_A, _ = self._get_weight(lora_A.data,
                                             'linear_qkv.lora_A.weight')
                lora_B, _ = self._get_weight(lora_B.data,
                                             'linear_qkv.lora_B.weight')

                if lora_A is not None:
                    self._peft_target_modules.update(
                        {'q_proj', 'k_proj', 'v_proj'})
                    # Split lora_B for Q, K, V
                    for key in ['q_proj', 'k_proj', 'v_proj']:
                        yield f'{prefix}self_attn.{key}.lora_A.weight', lora_A.clone(
                        )

                    num_kv_heads = self.config.num_key_value_heads
                    head_dim = self.config.hidden_size // self.config.num_attention_heads
                    heads_per_group = self.config.num_attention_heads // num_kv_heads
                    q_dim = heads_per_group * head_dim

                    lora_B = lora_B.reshape(num_kv_heads, -1, lora_B.shape[-1])
                    yield f'{prefix}self_attn.q_proj.lora_B.weight', lora_B[:, :q_dim, :].reshape(
                        -1, lora_B.shape[-1]).clone()
                    yield f'{prefix}self_attn.k_proj.lora_B.weight', lora_B[:,
                                                                            q_dim:
                                                                            -head_dim, :].reshape(
                                                                                -1,
                                                                                lora_B
                                                                                .
                                                                                shape[
                                                                                    -1]
                                                                            ).clone(
                                                                            )
                    yield f'{prefix}self_attn.v_proj.lora_B.weight', lora_B[:, -head_dim:, :].reshape(
                        -1, lora_B.shape[-1]).clone()

        # O projection LoRA
        if isinstance(mg_attn.linear_proj, LoraParallelLinear):
            lora_A = _deep_getattr(mg_attn.linear_proj,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = _deep_getattr(mg_attn.linear_proj,
                                  f'lora_B.{self._adapter_name}.weight')

            if lora_A is not None and lora_B is not None:
                lora_A, _ = self._get_weight(lora_A.data,
                                             'linear_proj.lora_A.weight')
                lora_B, _ = self._get_weight(lora_B.data,
                                             'linear_proj.lora_B.weight')

                if lora_A is not None:
                    self._peft_target_modules.add('o_proj')
                    yield f'{prefix}self_attn.o_proj.lora_A.weight', lora_A.clone(
                    )
                    yield f'{prefix}self_attn.o_proj.lora_B.weight', lora_B.clone(
                    )

        # MLP LoRA
        if hasattr(mg_mlp, 'linear_fc1') and isinstance(
                mg_mlp.linear_fc1, LoraParallelLinear):
            lora_A = _deep_getattr(mg_mlp.linear_fc1,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = _deep_getattr(mg_mlp.linear_fc1,
                                  f'lora_B.{self._adapter_name}.weight')

            if lora_A is not None and lora_B is not None:
                lora_A, _ = self._get_weight(lora_A.data,
                                             'linear_fc1.lora_A.weight')
                lora_B, _ = self._get_weight(lora_B.data,
                                             'linear_fc1.lora_B.weight')

                if lora_A is not None:
                    self._peft_target_modules.update({'gate_proj', 'up_proj'})
                    for key in ['gate_proj', 'up_proj']:
                        yield f'{prefix}mlp.{key}.lora_A.weight', lora_A.clone(
                        )

                    lora_B = lora_B.reshape(2, -1, lora_B.shape[-1])
                    yield f'{prefix}mlp.gate_proj.lora_B.weight', lora_B[
                        0].clone()
                    yield f'{prefix}mlp.up_proj.lora_B.weight', lora_B[
                        1].clone()

        if hasattr(mg_mlp, 'linear_fc2') and isinstance(
                mg_mlp.linear_fc2, LoraParallelLinear):
            lora_A = _deep_getattr(mg_mlp.linear_fc2,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = _deep_getattr(mg_mlp.linear_fc2,
                                  f'lora_B.{self._adapter_name}.weight')

            if lora_A is not None and lora_B is not None:
                lora_A, _ = self._get_weight(lora_A.data,
                                             'linear_fc2.lora_A.weight')
                lora_B, _ = self._get_weight(lora_B.data,
                                             'linear_fc2.lora_B.weight')

                if lora_A is not None:
                    self._peft_target_modules.add('down_proj')
                    yield f'{prefix}mlp.down_proj.lora_A.weight', lora_A.clone(
                    )
                    yield f'{prefix}mlp.down_proj.lora_B.weight', lora_B.clone(
                    )

    def save_weights(
        self,
        mg_models: Union[nn.Module, List[nn.Module]],
        output_dir: str,
        is_peft_format: bool = False,
    ) -> None:
        """Save Megatron model weights in HuggingFace format.

        Args:
            mg_models: Megatron model(s) to save.
            output_dir: Directory to save weights.
            is_peft_format: Whether saving in PEFT format.

        Note:
            For DP > 1, only DP rank 0 writes to disk. All ranks participate
            in tensor gather operations for TP.
        """
        torch_util.empty_cache()

        # Determine if this rank should write
        should_write = _is_last_rank()

        # Only the writing rank creates the saver
        saver = None
        if should_write:
            saver = _StreamingSafetensorSaver(
                save_dir=output_dir,
                max_shard_size=self.config.max_shard_size,
                is_peft_format=is_peft_format,
            )

        # All ranks participate in export (needed for TP gather)
        for key, tensor in self.export_weights(
                mg_models,
                target_device='cpu',
                only_last_rank=True,
                is_peft_format=is_peft_format,
                tqdm_desc='Saving: ',
        ):
            if saver is not None and tensor is not None:
                saver.add_tensor(key, tensor)

        if saver is not None:
            saver.finalize()

        # Save config on writing rank only
        if should_write:
            if is_peft_format and not isinstance(mg_models, (list, tuple)):
                mg_models = [mg_models]

            if is_peft_format and hasattr(mg_models[0], 'peft_config'):
                peft_config = copy(mg_models[0].peft_config.get(
                    self._adapter_name))
                if peft_config is not None:
                    peft_config.target_modules = list(
                        self._peft_target_modules)
                    peft_config.modules_to_save = list(
                        self._peft_modules_to_save)
                    peft_config.save_pretrained(output_dir)
            elif not is_peft_format and self.hf_config is not None:
                # Save HF config
                self.hf_config.vocab_size = self.config.padded_vocab_size
                self.hf_config.save_pretrained(output_dir)

        # Synchronize all ranks before continuing
        if dist.is_initialized():
            dist.barrier()


class BridgeAdapter:
    """Adapter for weight loading using GPTBridge.

    Provides a simple interface for loading HF weights into Megatron models.
    """
    def __init__(
        self,
        hf_config: Any,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        etp_size: Optional[int] = None,
        model_path: Optional[str] = None,
        padded_vocab_size: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the bridge adapter."""
        self.hf_config = hf_config
        self.model_path = model_path

        # Create bridge config
        self.config = BridgeConfig.from_hf_config(
            hf_config=hf_config,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            padded_vocab_size=padded_vocab_size,
        )
        if etp_size is not None:
            self.config.etp_size = etp_size

        self._bridge = None

    def _get_bridge(self) -> GPTBridge:
        """Get or create the bridge instance."""
        if self._bridge is None:
            self._bridge = GPTBridge(
                config=self.config,
                hf_config=self.hf_config,
            )
        return self._bridge

    def load_weights(
        self,
        mg_model: nn.Module,
        model_path: Optional[str] = None,
        is_peft_format: bool = False,
        adapter_name: str = 'default',
    ) -> None:
        """Load HuggingFace weights into Megatron model."""
        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError('model_path must be provided')

        bridge = self._get_bridge()
        bridge.load_weights(mg_model, model_path, is_peft_format, adapter_name)

    def save_weights(
        self,
        mg_models: Union[nn.Module, List[nn.Module]],
        output_dir: str,
        is_peft_format: bool = False,
    ) -> None:
        """Save Megatron model weights in HuggingFace format."""
        bridge = self._get_bridge()
        bridge.save_weights(mg_models, output_dir, is_peft_format)


class BridgeInitializer:
    """
    Megatron model initializer.

    This class provides complete model initialization flow including:
    - Megatron parallel state initialization
    - Model creation from HuggingFace config
    - Weight loading using GPTBridge

    Example:
        initializer = BridgeInitializer(
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
        cp_size: int = 1,
        ep_size: int = 1,
        etp_size: Optional[int] = None,
        vpp_size: Optional[int] = None,
        params_dtype=None,
        seed: int = 42,
        use_cpu_initialization: bool = False,
        attention_backend: str = 'flash',
        sequence_parallel: bool = False,
        recompute_granularity: Optional[str] = 'selective',
        recompute_modules: Optional[list] = None,
        recompute_method: Optional[str] = None,
        recompute_num_layers: Optional[int] = None,
    ):
        """Initialize BridgeInitializer.

        Args:
            tp_size: Tensor parallel size.
            pp_size: Pipeline parallel size.
            cp_size: Context parallel size.
            ep_size: Expert parallel size.
            etp_size: Expert tensor parallel size.
            params_dtype: Parameter dtype (default: torch.bfloat16).
            use_cpu_initialization: Initialize on CPU first.
            attention_backend: Attention backend.
            sequence_parallel: Enable sequence parallelism. Required for MoE with TP > 1.
            recompute_granularity: Activation recomputation strategy.
                'selective' (default): Only recompute core attention (memory efficient).
                'full': Recompute entire transformer layer (most memory efficient).
                None: No recomputation (fastest but highest memory).
            recompute_modules: Modules to recompute when using 'selective' granularity.
                Default: ['core_attn'] for efficient memory/compute trade-off.
            recompute_method: Method for full recompute ('uniform' or 'block').
                Required when recompute_granularity='full'.
            recompute_num_layers: Number of layers to recompute for 'full' mode.
                Required when recompute_granularity='full'.
        """
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.ep_size = ep_size
        self.etp_size = etp_size or tp_size
        self.vpp_size = vpp_size or 1
        self.params_dtype = params_dtype if params_dtype is not None else torch.bfloat16
        self.use_cpu_initialization = use_cpu_initialization
        self.attention_backend = attention_backend
        self.sequence_parallel = sequence_parallel
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules or ['core_attn']
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers
        self.seed = seed
        self._model = None
        self._bridge = None
        self._hf_config = None
        self._model_path = None
        self._initialized = False
        self._parallel_state = None
        self.config = None

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return

        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        dist.init_process_group(backend='nccl')

        init_kwargs = {
            'tensor_model_parallel_size': self.tp_size,
            'pipeline_model_parallel_size': self.pp_size,
            'context_parallel_size': self.cp_size,
            'virtual_pipeline_model_parallel_size': self.vpp_size,
            'expert_model_parallel_size': self.ep_size,
        }

        if exists('megatron_core>=0.13'):
            init_kwargs['expert_tensor_parallel_size'] = self.etp_size
        init_kwargs.update(kwargs)
        parallel_state.initialize_model_parallel(**init_kwargs)
        model_parallel_cuda_manual_seed(self.seed)

        self._parallel_state = parallel_state
        self._initialized = True

    def _create_model_from_config(
        self,
        hf_config: Any,
        padded_vocab_size: int,
    ) -> List[nn.Module]:
        """Create Megatron GPT model from HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration.
            padded_vocab_size: Padded vocabulary size.

        Returns:
            Megatron GPT model.
        """
        from megatron.core import parallel_state as mpu
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.enums import AttnBackend
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec, )

        # Convert HF config to Megatron config
        from ..tuners.multi_lora import convert_hf_config
        mg_config_dict = convert_hf_config(hf_config)

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
            """Finalize model grads that handles both DDP and PEFT/LoRA models.

            For DDP-wrapped models: Delegates to Megatron's native finalize_model_grads
            For PEFT/LoRA models: Manually all-reduce gradients across DP ranks

            This is necessary because PEFT models don't have ddp_config attribute
            that Megatron's native implementation expects.
            """

            # Check if model is DDP-wrapped (has ddp_config)
            if hasattr(model[0], 'ddp_config'):
                # Use native implementation for DDP models
                return _native_finalize_model_grads(model, num_tokens,
                                                    pg_collection)

            # For PEFT/LoRA models, call finish_grad_sync on each chunk
            # The model should have finish_grad_sync added by MegatronModel.add_adapter_to_model
            for model_chunk in model:
                if hasattr(model_chunk, 'finish_grad_sync'):
                    model_chunk.finish_grad_sync()

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
            expert_model_parallel_size=self.ep_size,
            sequence_parallel=use_sequence_parallel,
            params_dtype=self.params_dtype,
            pipeline_dtype=self.
            params_dtype,  # Required when using pipeline parallelism
            use_cpu_initialization=self.use_cpu_initialization,
            add_qkv_bias=mg_config_dict.get('add_qkv_bias', False),
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
        rope_scaling = {}
        if hasattr(hf_config, 'rope_scaling') and hf_config.rope_scaling is not None and 'factor' in hf_config.rope_scaling:
            rope_scaling = {
                'seq_len_interpolation_factor': hf_config.rope_scaling["factor"]
            }
        if mpu.get_virtual_pipeline_model_parallel_world_size() > 1:
            model = []
            has_vp_stage = inspect.signature(mpu.is_pipeline_first_stage).parameters.get("vp_stage", None) is not None
            for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                extra_kwargs = {} if not has_vp_stage else {"ignore_virtual": False, "vp_stage": i}
                _model = GPTModel(
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
                    **rope_scaling
                )
                model.append(_model)
            mpu.set_virtual_pipeline_model_parallel_rank(0)
        else:
            model = GPTModel(
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
            )
            model = [model]
        return model

    def _pad_vocab_size(self, vocab_size: int) -> int:
        """Pad vocab size for tensor parallelism."""
        divisor = self.tp_size * 128
        return ((vocab_size + divisor - 1) // divisor) * divisor

    def create_model(
        self,
        model_path: str,
        load_weights: bool = True,
        **kwargs,
    ) -> List[nn.Module]:
        """Create Megatron model from HuggingFace checkpoint.

        Args:
            model_path: Path to HuggingFace model or model ID.
            load_weights: Whether to load weights.

        Returns:
            Megatron model.
        """
        from transformers import AutoConfig

        # Download model if needed
        model_path = HubOperation.download_model(model_path)
        self._model_path = model_path

        # Load HF config first (needed for initialization)
        self._hf_config = AutoConfig.from_pretrained(model_path,
                                                     trust_remote_code=True)

        # Initialize Megatron parallel state with hf_config for proper args setup
        self.initialize(**kwargs)

        # Calculate padded vocab size
        padded_vocab_size = self._pad_vocab_size(self._hf_config.vocab_size)

        # Create model
        self._model = self._create_model_from_config(self._hf_config,
                                                     padded_vocab_size)

        # Load weights
        if load_weights:
            bridge_adapter = BridgeAdapter(
                hf_config=self._hf_config,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                ep_size=self.ep_size,
                etp_size=self.etp_size,
                model_path=model_path,
                padded_vocab_size=padded_vocab_size,
            )
            for _model in self._model:
                bridge_adapter.load_weights(_model, model_path)
            self._bridge = bridge_adapter._get_bridge()

        # Synchronize all ranks after model creation and weight loading
        # This is critical for Pipeline Parallel to ensure all ranks are ready
        # before any collective communication operations
        if dist.is_initialized():
            dist.barrier()

        return self._model

    @property
    def hf_config(self):
        """Get the HuggingFace config."""
        return self._hf_config

    @property
    def bridge(self):
        """Get the bridge instance."""
        return self._bridge

    def load_weights(self, model: nn.Module, model_path: str):
        """Load weights into an existing model.

        Args:
            model: Megatron model.
            model_path: Path to HuggingFace checkpoint.
        """
        if self._bridge is None and self._hf_config is None:
            raise ValueError('Must call create_model first')

        padded_vocab_size = self._pad_vocab_size(self._hf_config.vocab_size)
        bridge_adapter = BridgeAdapter(
            hf_config=self._hf_config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            ep_size=self.ep_size,
            model_path=model_path,
            padded_vocab_size=padded_vocab_size,
        )
        bridge_adapter.load_weights(model, model_path)

    def save_weights(self,
                     models: Union[nn.Module, List[nn.Module]],
                     output_dir: str,
                     is_peft_format: bool = False):
        """Save weights in HuggingFace format.

        Args:
            models: Megatron model(s).
            output_dir: Output directory.
            is_peft_format: Whether to save in PEFT format.
        """
        if self._bridge is None:
            raise ValueError('Must load weights first')

        if not isinstance(models, (list, tuple)):
            models = [models]

        self._bridge.save_weights(models,
                                  output_dir,
                                  is_peft_format=is_peft_format)
