# Copyright (c) twinkle authors. All rights reserved.
# GPT Bridge for HuggingFace to Megatron-Core weight conversion.
import os
from copy import copy
from typing import Any, Generator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from twinkle.hub import HubOperation
from twinkle.utils import SafetensorLazyLoader, StreamingSafetensorSaver, deep_getattr
from twinkle.utils.platform import Platform
from twinkle.infra import is_last_rank
from twinkle.megatron.args import get_args, set_args, TwinkleArgs

try:
    from megatron.core import mpu
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    mpu = None

# deprecated
class TwinkleGPTBridge:
    """GPT Bridge for HuggingFace to Megatron-Core weight conversion.
    
    Uses global get_args() for configuration. The args should be set before
    creating the bridge instance via set_args(TwinkleArgs.from_hf_config(...)).
    """
    # HuggingFace model structure constants
    # LLM models
    HF_LAYERS_PREFIX = 'model.layers'
    HF_EMBED_KEY = 'model.embed_tokens.weight'
    HF_FINAL_LAYERNORM_KEY = 'model.norm.weight'
    HF_LM_HEAD_KEY = 'lm_head.weight'
    
    # MLLMs - language model is nested
    HF_VL_LAYERS_PREFIX = 'model.language_model.layers'
    HF_VL_EMBED_KEY = 'model.language_model.embed_tokens.weight'
    HF_VL_FINAL_LAYERNORM_KEY = 'model.language_model.norm.weight'
    HF_VL_LM_HEAD_KEY = 'model.language_model.lm_head.weight'

    def __init__(self,
                 hf_config: Any = None,
                 disable_tqdm: bool = False):
        """Initialize the bridge.

        Args:
            hf_config: HuggingFace model config (for VL model detection).
            disable_tqdm: Whether to disable progress bar.
        """
        # Get configuration from global args
        args = get_args()
        self.args = args
        self.hf_config = hf_config
        self.disable_tqdm = disable_tqdm or not is_last_rank()
        
        # Detect VL model for correct weight prefixes
        self._is_vl_model = args.is_multimodal
        if self._is_vl_model:
            self._layers_prefix = self.HF_VL_LAYERS_PREFIX
            self._embed_key = self.HF_VL_EMBED_KEY
            self._final_layernorm_key = self.HF_VL_FINAL_LAYERNORM_KEY
            self._lm_head_key = self.HF_VL_LM_HEAD_KEY
        else:
            self._layers_prefix = self.HF_LAYERS_PREFIX
            self._embed_key = self.HF_EMBED_KEY
            self._final_layernorm_key = self.HF_FINAL_LAYERNORM_KEY
            self._lm_head_key = self.HF_LM_HEAD_KEY

        # Parallel state from args
        self.tp_size = args.tp_size
        self.pp_size = args.pp_size
        self.ep_size = args.ep_size
        self.etp_size = args.etp_size

        # Get parallel ranks from Megatron
        if MEGATRON_AVAILABLE and mpu.is_initialized():
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
            raise ValueError("Megatron is not initialized. Please initialize megatron first.")

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

        if self._only_last_rank and not is_last_rank():
            return None, None

        return tensor, None

    # =========================================================================
    # Weight Loading Methods
    # =========================================================================

    def _load_embedding(self, mg_model, loader: SafetensorLazyLoader):
        """Load embedding weights."""
        embed_module = deep_getattr(mg_model, 'embedding.word_embeddings')
        if embed_module is None:
            return

        hf_weight = loader.get_tensor(self._embed_key)

        # Pad vocabulary if needed
        if hf_weight.shape[0] < self.args.padded_vocab_size:
            hf_weight = F.pad(
                hf_weight,
                (0, 0, 0, self.args.padded_vocab_size - hf_weight.shape[0]))

        self._set_weight(embed_module.weight, hf_weight,
                         'word_embeddings.weight')

    def _load_output_layer(self, mg_model, loader: SafetensorLazyLoader):
        """Load output layer (lm_head) weights."""
        output_module = deep_getattr(mg_model, 'output_layer')
        if output_module is None or output_module.weight is None:
            return

        # Check if weights are tied
        if self.args.tie_word_embeddings:
            hf_weight = loader.get_tensor(self._embed_key)
        else:
            hf_weight = loader.get_tensor(self._lm_head_key)

        # Pad vocabulary if needed
        if hf_weight.shape[0] < self.args.padded_vocab_size:
            hf_weight = F.pad(
                hf_weight,
                (0, 0, 0, self.args.padded_vocab_size - hf_weight.shape[0]))

        self._set_weight(output_module.weight, hf_weight,
                         'output_layer.weight')

    def _load_final_layernorm(self, mg_model, loader: SafetensorLazyLoader):
        """Load final layer norm weights."""
        ln_module = deep_getattr(mg_model, 'decoder.final_layernorm')
        if ln_module is None:
            return

        hf_weight = loader.get_tensor(self._final_layernorm_key)
        ln_module.weight.data.copy_(hf_weight)

    def _load_attention(self, mg_layer, loader: SafetensorLazyLoader,
                        layer_idx: int):
        """Load attention layer weights."""
        mg_attn = mg_layer.self_attention
        prefix = f'{self._layers_prefix}.{layer_idx}.self_attn.'

        num_heads = self.args.num_attention_heads
        num_kv_heads = self.args.num_key_value_heads
        hidden_size = self.args.hidden_size
        # Use kv_channels (head_dim) from config if available (for Qwen3 etc.)
        head_dim = getattr(self.args, 'kv_channels',
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
        if self.args.add_qkv_bias:
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
        ln_key = f'{self._layers_prefix}.{layer_idx}.input_layernorm.weight'
        ln_weight = loader.get_tensor(ln_key)

        ln_param = deep_getattr(mg_attn, 'linear_qkv.layer_norm_weight')
        if ln_param is not None:
            ln_param.data.copy_(ln_weight)
        else:
            ln_module = deep_getattr(mg_layer, 'input_layernorm')
            if ln_module is not None:
                ln_module.weight.data.copy_(ln_weight)

        # QK layernorm (Qwen3)
        if self.args.qk_layernorm:
            try:
                q_norm = loader.get_tensor(f'{prefix}q_norm.weight')
                k_norm = loader.get_tensor(f'{prefix}k_norm.weight')
                q_ln = deep_getattr(mg_attn, 'q_layernorm')
                k_ln = deep_getattr(mg_attn, 'k_layernorm')
                if q_ln is not None:
                    q_ln.weight.data.copy_(q_norm)
                if k_ln is not None:
                    k_ln.weight.data.copy_(k_norm)
            except KeyError:
                pass

    def _load_mlp(self, mg_layer, loader: SafetensorLazyLoader, layer_idx: int):
        """Load MLP layer weights."""
        mg_mlp = mg_layer.mlp
        prefix = f'{self._layers_prefix}.{layer_idx}.mlp.'

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
        ln_key = f'{self._layers_prefix}.{layer_idx}.post_attention_layernorm.weight'
        try:
            ln_weight = loader.get_tensor(ln_key)

            ln_param = deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
            if ln_param is not None:
                ln_param.data.copy_(ln_weight)
            else:
                ln_module = deep_getattr(mg_layer, 'pre_mlp_layernorm')
                if ln_module is not None:
                    ln_module.weight.data.copy_(ln_weight)
        except KeyError:
            pass

    def _load_moe(self, mg_layer, loader: SafetensorLazyLoader, layer_idx: int):
        """Load MoE layer weights.

        Handles Expert Parallel (EP) sharding - each EP rank loads only its
        assigned subset of experts based on ep_rank and ep_size.

        For EP=2 with 128 experts:
          - EP rank 0 loads experts 0-63
          - EP rank 1 loads experts 64-127
        """
        mg_mlp = mg_layer.mlp
        prefix = f'{self._layers_prefix}.{layer_idx}.mlp.'

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
                router_module = deep_getattr(mg_mlp, 'router')
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
        if self.args.shared_expert_intermediate_size > 0:
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

                    shared_module = deep_getattr(mg_mlp, 'shared_experts')
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
                        shared_module = deep_getattr(mg_mlp, 'shared_experts')
                        if shared_module is not None and hasattr(
                                shared_module, 'gate_weight'):
                            shared_module.gate_weight.data.copy_(gate_weight)
                        break
                    except KeyError:
                        continue

        # Load experts with EP sharding
        num_local_experts = self.args.num_experts // self.ep_size
        start_expert_idx = self.ep_rank * num_local_experts
        experts_module = deep_getattr(mg_mlp, 'experts')

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
        ln_key = f'{self._layers_prefix}.{layer_idx}.post_attention_layernorm.weight'
        try:
            ln_weight = loader.get_tensor(ln_key)
            # Try pre_mlp_layernorm first (used in MoE layers)
            ln_module = deep_getattr(mg_layer, 'pre_mlp_layernorm')
            if ln_module is not None and hasattr(ln_module, 'weight'):
                ln_module.weight.data.copy_(ln_weight)
            else:
                # Fallback to linear_fc1.layer_norm_weight
                ln_param = deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
                if ln_param is not None:
                    ln_param.data.copy_(ln_weight)
        except KeyError:
            pass

    def _load_layer(self, mg_layer, loader: SafetensorLazyLoader, layer_idx: int):
        """Load a single transformer layer."""
        self._load_attention(mg_layer, loader, layer_idx)

        # Check if MoE layer
        if self.args.num_experts > 0:
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
            with SafetensorLazyLoader(model_path, is_peft_format=is_peft_format) as loader:
                if is_peft_format:
                    self._load_peft_weights(mg_model, loader)
                else:
                    self._load_base_weights(mg_model, loader)

    def _load_base_weights(self, mg_model: nn.Module,
                           loader: SafetensorLazyLoader):
        """Load base model weights."""
        # Get decoder
        decoder = deep_getattr(mg_model, 'decoder')
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
                           loader: SafetensorLazyLoader):
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
        decoder = deep_getattr(mg_model, 'decoder')
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
                target = deep_getattr(mg_attn, 'linear_qkv')
            elif 'o_proj' in key:
                target = deep_getattr(mg_attn, 'linear_proj')
            else:
                return
        elif 'mlp' in key:
            mg_mlp = mg_layer.mlp
            if 'gate_proj' in key or 'up_proj' in key:
                target = deep_getattr(mg_mlp, 'linear_fc1')
            elif 'down_proj' in key:
                target = deep_getattr(mg_mlp, 'linear_fc2')
            else:
                return
        else:
            return

        if target is None:
            return

        # Get LoRA module
        if is_lora_A:
            lora_module = deep_getattr(target, f'lora_A.{self._adapter_name}')
        else:
            lora_module = deep_getattr(target, f'lora_B.{self._adapter_name}')

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

            decoder = deep_getattr(mg_model, 'decoder')
            if decoder is None:
                decoder = mg_model

            layers = getattr(decoder, 'layers', [])

            if not is_peft_format:
                # Export embedding
                if self.pp_size <= 1 or self.pp_rank == 0:
                    embed = deep_getattr(mg_model,
                                         'embedding.word_embeddings.weight')
                    if embed is not None:
                        weight, _ = self._get_weight(embed.data,
                                                     'word_embeddings.weight')
                        if weight is not None:
                            weight = weight[:self.args.vocab_size]
                            yield f'{hf_prefix}{self._embed_key}', weight

            # Export layers
            prog_bar = tqdm(layers, desc=tqdm_desc, disable=self.disable_tqdm)
            for mg_layer in prog_bar:
                layer_idx = mg_layer.layer_number - 1
                yield from self._export_layer(mg_layer, layer_idx, hf_prefix,
                                              is_peft_format)

            if not is_peft_format:
                # Export final layernorm and output layer
                if self.pp_size <= 1 or self.pp_rank == self.pp_size - 1:
                    ln_module = deep_getattr(mg_model,
                                             'decoder.final_layernorm')
                    if ln_module is not None:
                        yield f'{hf_prefix}{self._final_layernorm_key}', ln_module.weight.data.clone(
                        )

                    output = deep_getattr(mg_model, 'output_layer.weight')
                    if output is not None:
                        weight, _ = self._get_weight(output.data,
                                                     'output_layer.weight')
                        if weight is not None:
                            weight = weight[:self.args.vocab_size]
                            yield f'{hf_prefix}{self._lm_head_key}', weight

    def _export_layer(
        self,
        mg_layer,
        layer_idx: int,
        hf_prefix: str,
        is_peft_format: bool,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Export a single layer."""
        prefix = f'{hf_prefix}{self._layers_prefix}.{layer_idx}.'

        mg_attn = mg_layer.self_attention
        mg_mlp = mg_layer.mlp

        num_heads = self.args.num_attention_heads
        num_kv_heads = self.args.num_key_value_heads
        hidden_size = self.args.hidden_size
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
            ln = deep_getattr(mg_attn, 'linear_qkv.layer_norm_weight')
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

            ln2 = deep_getattr(mg_mlp, 'linear_fc1.layer_norm_weight')
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
        from twinkle.megatron.tuners import LoraParallelLinear

        # Attention LoRA
        if isinstance(mg_attn.linear_qkv, LoraParallelLinear):
            lora_A = deep_getattr(mg_attn.linear_qkv,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = deep_getattr(mg_attn.linear_qkv,
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

                    num_kv_heads = self.args.num_key_value_heads
                    head_dim = self.args.hidden_size // self.args.num_attention_heads
                    heads_per_group = self.args.num_attention_heads // num_kv_heads
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
            lora_A = deep_getattr(mg_attn.linear_proj,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = deep_getattr(mg_attn.linear_proj,
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
            lora_A = deep_getattr(mg_mlp.linear_fc1,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = deep_getattr(mg_mlp.linear_fc1,
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
            lora_A = deep_getattr(mg_mlp.linear_fc2,
                                  f'lora_A.{self._adapter_name}.weight')
            lora_B = deep_getattr(mg_mlp.linear_fc2,
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
        torch.cuda.empty_cache()

        # Determine if this rank should write
        should_write = is_last_rank()

        # Only the writing rank creates the saver
        saver = None
        if should_write:
            saver = StreamingSafetensorSaver(
                save_dir=output_dir,
                max_shard_size=self.args.max_shard_size,
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
                if hasattr(self.hf_config, 'text_config') and hasattr(self.hf_config.text_config, 'vocab_size'):
                    self.hf_config.text_config.vocab_size = self.args.padded_vocab_size
                if hasattr(self.hf_config, 'vocab_size'):
                    self.hf_config.vocab_size = self.args.padded_vocab_size
                self.hf_config.save_pretrained(output_dir)

        # Synchronize all ranks before continuing
        if dist.is_initialized():
            dist.barrier()


class TwinkleBridgeAdapter:
    """Adapter for weight loading using TwinkleGPTBridge.

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

        # Create and set global args
        etp_size = etp_size or tp_size
        args = TwinkleArgs.from_hf_config(
            hf_config=hf_config,
            model_dir=model_path or '',
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            etp_size=etp_size,
            padded_vocab_size=padded_vocab_size,
        )
        set_args(args)
        self.args = args
        self._bridge = None

    def _get_bridge(self) -> TwinkleGPTBridge:
        """Get or create the bridge instance."""
        if self._bridge is None:
            self._bridge = TwinkleGPTBridge(hf_config=self.hf_config)
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


class TwinkleBridgeInitializer:
    """
    Megatron model initializer.

    This class provides complete model initialization flow including:
    - Megatron parallel state initialization
    - Model creation from HuggingFace config
    - Weight loading using TwinkleGPTBridge

    Example:
        initializer = TwinkleBridgeInitializer(
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
        params_dtype=None,
        use_cpu_initialization: bool = False,
        attention_backend: str = 'flash',
        sequence_parallel: bool = False,
        recompute_granularity: Optional[str] = 'selective',
        recompute_modules: Optional[list] = None,
        recompute_method: Optional[str] = None,
        recompute_num_layers: Optional[int] = None,
    ):
        """Initialize TwinkleBridgeInitializer.

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
        self.params_dtype = params_dtype if params_dtype is not None else torch.bfloat16
        self.use_cpu_initialization = use_cpu_initialization
        self.attention_backend = attention_backend
        self.sequence_parallel = sequence_parallel
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules or ['core_attn']
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers

        self._model = None
        self._bridge = None
        self._hf_config = None
        self._model_path = None

    def _download_model(self, model_path: str) -> str:
        """Download model if it's a model ID."""
        if os.path.isdir(model_path):
            return model_path

        try:
            from modelscope import snapshot_download
            return snapshot_download(model_path)
        except ImportError:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_path)

    def _initialize_megatron(self, hf_config: Any = None):
        """Initialize Megatron parallel state.

        This sets up the required process groups for tensor, pipeline,
        and data parallelism using Megatron's parallel state module directly.

        Handles both local (torchrun) and Ray execution modes:
        - Local: Uses torchrun's environment variables (already set)
        - Ray: Uses RayHelper's environment variables (RANK, WORLD_SIZE, etc.)

        Args:
            hf_config: Optional HuggingFace config for additional model parameters.
        """
        import os
        import torch.distributed as dist
        from datetime import timedelta
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        # Check if already initialized
        try:
            if mpu.is_initialized():
                return
        except AssertionError:
            pass

        # Determine execution mode
        twinkle_mode = os.environ.get('TWINKLE_MODE', 'local')

        # Initialize distributed if not already
        if not dist.is_initialized():
            if twinkle_mode == 'ray':
                # Ray mode: use environment variables set by RayHelper
                rank = int(os.environ.get('RANK', '0'))
                world_size = int(os.environ.get('WORLD_SIZE', '1'))
                master_addr = os.environ.get('MASTER_ADDR', 'localhost')
                master_port = os.environ.get('MASTER_PORT', '29500')
                local_rank = int(os.environ.get('LOCAL_RANK', '0'))

                # Set CUDA device before init_process_group
                torch.cuda.set_device(local_rank)

                # Initialize process group with explicit parameters
                dist.init_process_group(
                    backend='nccl',
                    init_method=f'tcp://{master_addr}:{master_port}',
                    rank=rank,
                    world_size=world_size,
                    timeout=timedelta(minutes=10),
                )
            else:
                # Local mode (torchrun): environment variables are already set
                dist.init_process_group(backend='nccl')

        # Initialize Megatron parallel state directly
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            context_parallel_size=self.cp_size,
            expert_model_parallel_size=self.ep_size,
        )

        # Initialize CUDA RNG tracker for tensor parallel random states
        # This is required when use_cpu_initialization=False (GPU initialization)
        model_parallel_cuda_manual_seed(42)

    def _create_model_from_config(
        self,
        hf_config: Any,
        padded_vocab_size: int,
    ) -> nn.Module:
        """Create Megatron GPT model from HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration.
            padded_vocab_size: Padded vocabulary size.

        Returns:
            Megatron GPT model.
        """
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu
        from megatron.core.transformer import TransformerConfig
        from megatron.core.transformer.enums import AttnBackend
        from megatron.core.models.gpt import GPTModel
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec, )

        # Convert HF config to Megatron config
        from ..utils import convert_hf_config
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
            from megatron.core import parallel_state as mpu

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
        self._transformer_config = config

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

        return model

    def _pad_vocab_size(self, vocab_size: int) -> int:
        """Pad vocab size for tensor parallelism."""
        divisor = self.tp_size * 128
        return ((vocab_size + divisor - 1) // divisor) * divisor

    def _get_vocab_size(self, hf_config) -> int:
        """Get vocab size from HuggingFace config.

        Handles both regular LLM configs and multimodal configs where
        vocab_size might be in text_config sub-config.

        Args:
            hf_config: HuggingFace model config.

        Returns:
            Vocabulary size.
        """
        # Try direct vocab_size first
        if hasattr(hf_config, 'vocab_size') and hf_config.vocab_size is not None:
            return hf_config.vocab_size

        # For multimodal models like Qwen3-VL, vocab_size is in text_config
        if hasattr(hf_config, 'text_config') and hasattr(hf_config.text_config, 'vocab_size'):
            return hf_config.text_config.vocab_size

        # Fallback for other multimodal configs
        for attr in ['llm_config', 'language_config', 'lm_config']:
            if hasattr(hf_config, attr):
                sub_config = getattr(hf_config, attr)
                if hasattr(sub_config, 'vocab_size'):
                    return sub_config.vocab_size

        raise ValueError(
            f'Cannot find vocab_size in config of type {type(hf_config).__name__}. '
            'Please check the config structure or add support for this model type.'
        )

    def create_model(
        self,
        model_path: str,
        load_weights: bool = True,
    ) -> nn.Module:
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
        self._initialize_megatron(self._hf_config)

        # Calculate padded vocab size
        vocab_size = self._get_vocab_size(self._hf_config)
        padded_vocab_size = self._pad_vocab_size(vocab_size)
        
        args = get_args()
        if args is None:
            args = TwinkleArgs.from_hf_config(
                hf_config=self._hf_config,
                model_dir=model_path,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                cp_size=self.cp_size,
                ep_size=self.ep_size,
                etp_size=self.etp_size,
                sequence_parallel=self.sequence_parallel,
                torch_dtype=self.params_dtype,
                padded_vocab_size=padded_vocab_size,
            )
            set_args(args)
        self._args = args

        # Create model
        self._model = self._create_model_from_config(self._hf_config,
                                                     padded_vocab_size)

        # Load weights
        if load_weights:
            bridge_adapter = TwinkleBridgeAdapter(
                hf_config=self._hf_config,
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                ep_size=self.ep_size,
                etp_size=self.etp_size,
                model_path=model_path,
                padded_vocab_size=padded_vocab_size,
            )
            bridge_adapter.load_weights(self._model, model_path)
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

        vocab_size = self._get_vocab_size(self._hf_config)
        padded_vocab_size = self._pad_vocab_size(vocab_size)
        bridge_adapter = TwinkleBridgeAdapter(
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



def load_hf_weights_to_megatron(
    mg_model: nn.Module,
    model_path: str,
    hf_config: Any,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    padded_vocab_size: Optional[int] = None,
) -> None:
    """Convenience function to load HF weights into Megatron model."""
    adapter = TwinkleBridgeAdapter(
        hf_config=hf_config,
        tp_size=tp_size,
        pp_size=pp_size,
        ep_size=ep_size,
        model_path=model_path,
        padded_vocab_size=padded_vocab_size,
    )
    adapter.load_weights(mg_model, model_path)
