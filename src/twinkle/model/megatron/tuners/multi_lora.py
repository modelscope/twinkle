# Copyright (c) ModelScope Contributors. All rights reserved.
"""Utility functions for Megatron-Core integration."""
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import threading

import torch
import torch.nn as nn
import torch.distributed as dist

import megatron.core
from megatron.core import parallel_state as mpu
from megatron.core.extensions.transformer_engine import (
    TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
)
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from packaging import version
from peft import LoraConfig, get_peft_model

from twinkle import Platform

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


# Config mapping from HuggingFace to Megatron
CONFIG_MAPPING = {
    'num_layers': ['num_hidden_layers'],
    'hidden_size': ['hidden_size'],
    'mlp_ffn_hidden_size': ['intermediate_size_mlp'],
    'ffn_hidden_size': ['intermediate_size'],
    'num_attention_heads': ['num_attention_heads'],
    'num_query_groups': ['num_key_value_heads'],
    'max_position_embeddings': ['max_position_embeddings'],
    'norm_epsilon': ['rms_norm_eps'],
    'rotary_base': ['rope_theta'],
    'padded_vocab_size': ['vocab_size'],
    'attention_dropout': ['attention_dropout'],
    'untie_embeddings_and_output_weights': ['tie_word_embeddings'],
    'swiglu': ['hidden_act'],
    'add_qkv_bias': ['attention_bias', 'qkv_bias', 'use_bias'],
    'disable_bias_linear': ['mlp_bias'],
    'kv_channels': ['head_dim', 'v_head_dim'],
    'architectures': ['architectures'],
    # moe
    'moe_ffn_hidden_size': ['moe_intermediate_size'],
    'moe_shared_expert_intermediate_size': ['shared_expert_intermediate_size'],
    'moe_router_topk': ['num_experts_per_tok', 'moe_topk', 'moe_k'],
    'moe_router_num_groups': ['n_group'],
    'moe_router_group_topk': ['topk_group'],
    'num_experts': ['num_experts', 'n_routed_experts', 'moe_num_experts', 'num_local_experts'],
    'moe_router_pre_softmax': ['norm_topk_prob'],
    # deepseek
    'q_lora_rank': ['q_lora_rank'],
    'kv_lora_rank': ['kv_lora_rank'],
    'moe_router_score_function': ['scoring_func'],
    'moe_router_bias_update_rate': ['aux_loss_alpha'],
    'qk_head_dim': ['qk_nope_head_dim'],
    'qk_pos_emb_head_dim': ['qk_rope_head_dim'],
    'moe_router_topk_scaling_factor': ['routed_scaling_factor'],
    'qk_layernorm': ['use_qk_norm'],
    # other
    'original_max_position_embeddings': ['original_max_position_embeddings'],
    'partial_rotary_factor': ['partial_rotary_factor'],
    'first_k_dense_replace': ['first_k_dense_replace', 'moe_layer_start_index'],
    'n_shared_experts': ['n_shared_experts', 'num_shared_expert', 'moe_num_shared_experts'],
    'window_size': ['sliding_window'],
    'layer_types': ['layer_types'],
}


class TenantProcessGroupManager:
    """Manager for multi-tenant process groups.
    
    In a multi-tenant scenario, multiple users may share the same base model in a single
    process, each with their own LoRA adapters. To avoid communication interference between
    tenants, we need to maintain separate process groups for each tenant.
    
    This class provides:
    1. Per-tenant process group isolation
    2. Context managers to temporarily switch active process groups
    3. Patching of Megatron's communication operations to use tenant-specific groups
    
    Example:
        # Create tenant-specific groups
        manager = TenantProcessGroupManager()
        manager.register_tenant('user_1', tp_ranks=[0, 1], dp_ranks=[0, 2])
        manager.register_tenant('user_2', tp_ranks=[2, 3], dp_ranks=[1, 3])
        
        # Use tenant context for operations
        with manager.tenant_context('user_1'):
            # All Megatron communications will use user_1's process groups
            model.forward(...)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Tenant ID -> Process Groups mapping
        self._tenant_groups: Dict[str, Dict[str, dist.ProcessGroup]] = {}
        # Current active tenant (thread-local)
        self._active_tenant = threading.local()
        # Original Megatron parallel state functions (for patching)
        self._original_functions = {}
        # Whether patching is active
        self._patched = False
        
    def register_tenant(
        self,
        tenant_id: str,
        tp_ranks: Optional[List[int]] = None,
        pp_ranks: Optional[List[int]] = None,
        dp_ranks: Optional[List[int]] = None,
        ep_ranks: Optional[List[int]] = None,
        cp_ranks: Optional[List[int]] = None,
    ) -> None:
        """Register a tenant with specific process group ranks.
        
        Args:
            tenant_id: Unique identifier for the tenant.
            tp_ranks: Ranks for tensor parallel group.
            pp_ranks: Ranks for pipeline parallel group.
            dp_ranks: Ranks for data parallel group.
            ep_ranks: Ranks for expert parallel group.
            cp_ranks: Ranks for context parallel group.
        """
        if tenant_id in self._tenant_groups:
            return  # Already registered
            
        groups = {}
        
        # Create process groups for each parallelism dimension
        if tp_ranks and len(tp_ranks) > 1:
            groups['tp'] = dist.new_group(tp_ranks)
        if pp_ranks and len(pp_ranks) > 1:
            groups['pp'] = dist.new_group(pp_ranks)
        if dp_ranks and len(dp_ranks) > 1:
            groups['dp'] = dist.new_group(dp_ranks)
        if ep_ranks and len(ep_ranks) > 1:
            groups['ep'] = dist.new_group(ep_ranks)
        if cp_ranks and len(cp_ranks) > 1:
            groups['cp'] = dist.new_group(cp_ranks)
            
        self._tenant_groups[tenant_id] = groups
        
    def unregister_tenant(self, tenant_id: str) -> None:
        """Unregister a tenant and destroy its process groups.
        
        Args:
            tenant_id: Tenant to unregister.
        """
        if tenant_id in self._tenant_groups:
            groups = self._tenant_groups.pop(tenant_id)
            for group in groups.values():
                dist.destroy_process_group(group)
                
    def get_tenant_group(self, tenant_id: str, group_type: str) -> Optional[dist.ProcessGroup]:
        """Get process group for a tenant.
        
        Args:
            tenant_id: Tenant identifier.
            group_type: Type of group ('tp', 'pp', 'dp', 'ep', 'cp').
            
        Returns:
            Process group or None if not found.
        """
        if tenant_id in self._tenant_groups:
            return self._tenant_groups[tenant_id].get(group_type)
        return None
        
    @property
    def active_tenant(self) -> Optional[str]:
        """Get the currently active tenant ID."""
        return getattr(self._active_tenant, 'id', None)
        
    @contextmanager
    def tenant_context(self, tenant_id: str):
        """Context manager to set active tenant for communications.
        
        All Megatron communication operations within this context will use
        the tenant-specific process groups. This includes:
        
        - Tensor Parallel (TP): get_tensor_model_parallel_group/rank/world_size
        - Data Parallel (DP): get_data_parallel_group/rank/world_size  
        - Pipeline Parallel (PP): get_pipeline_model_parallel_group/rank/world_size,
                                  is_pipeline_first_stage, is_pipeline_last_stage
        - Expert Parallel (EP): get_expert_model_parallel_group/rank/world_size
        - Context Parallel (CP): get_context_parallel_group/rank/world_size
        
        Args:
            tenant_id: Tenant to activate.
            
        Example:
            manager = get_tenant_manager()
            manager.register_tenant('user_1', tp_ranks=[0, 1], dp_ranks=[0, 2])
            
            with manager.tenant_context('user_1'):
                # All Megatron communications use user_1's groups
                output = model.forward(input_ids)
        """
        old_tenant = self.active_tenant
        self._active_tenant.id = tenant_id
        
        # Apply all patches if not already done
        if not self._patched:
            self._patch_megatron_parallel_state()
            self._patch_tensor_parallel_comms()
            self._patch_expert_parallel_comms()
            self._patch_context_parallel_comms()
            
        try:
            yield
        finally:
            self._active_tenant.id = old_tenant
            
    def _patch_megatron_parallel_state(self) -> None:
        """Patch Megatron's parallel_state to use tenant-specific groups.
        
        This patches the following functions for full TP/PP/DP/EP/CP support:
        - get_tensor_model_parallel_group / get_tensor_model_parallel_world_size / get_tensor_model_parallel_rank
        - get_data_parallel_group / get_data_parallel_world_size / get_data_parallel_rank
        - get_pipeline_model_parallel_group / get_pipeline_model_parallel_world_size / get_pipeline_model_parallel_rank
        - get_expert_model_parallel_group / get_expert_model_parallel_world_size / get_expert_model_parallel_rank
        - get_context_parallel_group / get_context_parallel_world_size / get_context_parallel_rank
        """
        if self._patched:
            return
            
        # Save original functions
        self._original_functions = {
            # TP functions
            'get_tensor_model_parallel_group': mpu.get_tensor_model_parallel_group,
            'get_tensor_model_parallel_world_size': mpu.get_tensor_model_parallel_world_size,
            'get_tensor_model_parallel_rank': mpu.get_tensor_model_parallel_rank,
            # DP functions
            'get_data_parallel_group': mpu.get_data_parallel_group,
            'get_data_parallel_world_size': mpu.get_data_parallel_world_size,
            'get_data_parallel_rank': mpu.get_data_parallel_rank,
            # PP functions
            'get_pipeline_model_parallel_group': mpu.get_pipeline_model_parallel_group,
            'get_pipeline_model_parallel_world_size': mpu.get_pipeline_model_parallel_world_size,
            'get_pipeline_model_parallel_rank': mpu.get_pipeline_model_parallel_rank,
            'is_pipeline_first_stage': mpu.is_pipeline_first_stage,
            'is_pipeline_last_stage': mpu.is_pipeline_last_stage,
            # EP functions
            'get_expert_model_parallel_group': mpu.get_expert_model_parallel_group,
            'get_expert_model_parallel_world_size': mpu.get_expert_model_parallel_world_size,
            'get_expert_model_parallel_rank': mpu.get_expert_model_parallel_rank,
            # CP functions
            'get_context_parallel_group': mpu.get_context_parallel_group,
            'get_context_parallel_world_size': mpu.get_context_parallel_world_size,
            'get_context_parallel_rank': mpu.get_context_parallel_rank,
        }
        
        manager = self
        
        def _make_group_getter(group_type: str, original_func_name: str):
            """Create patched group getter function."""
            def patched_func(*args, **kwargs):
                tenant = manager.active_tenant
                if tenant and tenant in manager._tenant_groups:
                    group = manager.get_tenant_group(tenant, group_type)
                    if group is not None:
                        return group
                return manager._original_functions[original_func_name](*args, **kwargs)
            return patched_func
        
        def _make_world_size_getter(group_type: str, original_func_name: str):
            """Create patched world_size getter function."""
            def patched_func(*args, **kwargs):
                tenant = manager.active_tenant
                if tenant and tenant in manager._tenant_groups:
                    group = manager.get_tenant_group(tenant, group_type)
                    if group is not None:
                        return dist.get_world_size(group)
                return manager._original_functions[original_func_name](*args, **kwargs)
            return patched_func
        
        def _make_rank_getter(group_type: str, original_func_name: str):
            """Create patched rank getter function."""
            def patched_func(*args, **kwargs):
                tenant = manager.active_tenant
                if tenant and tenant in manager._tenant_groups:
                    group = manager.get_tenant_group(tenant, group_type)
                    if group is not None:
                        return dist.get_rank(group)
                return manager._original_functions[original_func_name](*args, **kwargs)
            return patched_func
        
        # Apply patches for TP
        mpu.get_tensor_model_parallel_group = _make_group_getter('tp', 'get_tensor_model_parallel_group')
        mpu.get_tensor_model_parallel_world_size = _make_world_size_getter('tp', 'get_tensor_model_parallel_world_size')
        mpu.get_tensor_model_parallel_rank = _make_rank_getter('tp', 'get_tensor_model_parallel_rank')
        
        # Apply patches for DP
        mpu.get_data_parallel_group = _make_group_getter('dp', 'get_data_parallel_group')
        mpu.get_data_parallel_world_size = _make_world_size_getter('dp', 'get_data_parallel_world_size')
        mpu.get_data_parallel_rank = _make_rank_getter('dp', 'get_data_parallel_rank')
        
        # Apply patches for PP
        mpu.get_pipeline_model_parallel_group = _make_group_getter('pp', 'get_pipeline_model_parallel_group')
        mpu.get_pipeline_model_parallel_world_size = _make_world_size_getter('pp', 'get_pipeline_model_parallel_world_size')
        mpu.get_pipeline_model_parallel_rank = _make_rank_getter('pp', 'get_pipeline_model_parallel_rank')
        
        # Patch is_pipeline_first/last_stage
        def patched_is_pipeline_first_stage(*args, **kwargs):
            tenant = manager.active_tenant
            if tenant and tenant in manager._tenant_groups:
                group = manager.get_tenant_group(tenant, 'pp')
                if group is not None:
                    return dist.get_rank(group) == 0
            return manager._original_functions['is_pipeline_first_stage'](*args, **kwargs)
            
        def patched_is_pipeline_last_stage(*args, **kwargs):
            tenant = manager.active_tenant
            if tenant and tenant in manager._tenant_groups:
                group = manager.get_tenant_group(tenant, 'pp')
                if group is not None:
                    return dist.get_rank(group) == dist.get_world_size(group) - 1
            return manager._original_functions['is_pipeline_last_stage'](*args, **kwargs)
        
        mpu.is_pipeline_first_stage = patched_is_pipeline_first_stage
        mpu.is_pipeline_last_stage = patched_is_pipeline_last_stage
        
        # Apply patches for EP
        mpu.get_expert_model_parallel_group = _make_group_getter('ep', 'get_expert_model_parallel_group')
        mpu.get_expert_model_parallel_world_size = _make_world_size_getter('ep', 'get_expert_model_parallel_world_size')
        mpu.get_expert_model_parallel_rank = _make_rank_getter('ep', 'get_expert_model_parallel_rank')
        
        # Apply patches for CP
        mpu.get_context_parallel_group = _make_group_getter('cp', 'get_context_parallel_group')
        mpu.get_context_parallel_world_size = _make_world_size_getter('cp', 'get_context_parallel_world_size')
        mpu.get_context_parallel_rank = _make_rank_getter('cp', 'get_context_parallel_rank')
        
        self._patched = True
        
    def unpatch_megatron_parallel_state(self) -> None:
        """Restore original Megatron parallel_state functions."""
        if not self._patched:
            return
            
        for name, func in self._original_functions.items():
            setattr(mpu, name, func)
            
        self._patched = False
        self._original_functions = {}
        
    def _patch_tensor_parallel_comms(self) -> None:
        """Patch tensor parallel communication operations.
        
        This patches critical TP communication functions in megatron.core.tensor_parallel:
        - mappings.copy_to_tensor_model_parallel_region
        - mappings.reduce_from_tensor_model_parallel_region
        - mappings.scatter_to_tensor_model_parallel_region
        - mappings.gather_from_tensor_model_parallel_region
        """
        try:
            from megatron.core.tensor_parallel import mappings
        except ImportError:
            return
            
        if hasattr(self, '_tp_comms_patched') and self._tp_comms_patched:
            return
            
        # Save original functions
        self._original_tp_functions = {}
        
        # The mappings module uses get_tensor_model_parallel_group() internally,
        # which we've already patched. No additional patches needed here.
        # The patched group getters will be used automatically.
        
        self._tp_comms_patched = True
        
    def _patch_expert_parallel_comms(self) -> None:
        """Patch expert parallel communication operations for MoE models.
        
        For MoE models, expert parallel communications use:
        - get_expert_model_parallel_group
        - get_expert_tensor_parallel_group (if using expert tensor parallelism)
        
        Since we've patched the group getters, the communications will
        automatically use tenant-specific groups.
        """
        # Expert parallel communications use the patched group getters
        # No additional patches needed
        pass
        
    def _patch_context_parallel_comms(self) -> None:
        """Patch context parallel communication operations.
        
        Context parallelism communications include:
        - Ring attention communications
        - CP all-to-all operations
        
        These use get_context_parallel_group() which we've patched.
        """
        # CP communications use the patched group getters
        # No additional patches needed
        pass


# Global instance for easy access
_tenant_manager: Optional[TenantProcessGroupManager] = None


def get_tenant_manager() -> TenantProcessGroupManager:
    """Get the global tenant process group manager.
    
    
    Returns:
        The singleton TenantProcessGroupManager instance.
    """
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantProcessGroupManager()
    return _tenant_manager

def find_layers(model: nn.Module, cond_fn) -> List[str]:
    """Find all layers in model matching condition function.
    
    
    
    Args:
        model: The model to search.
        cond_fn: Callable(name, module) -> bool.
        
    Returns:
        List of matching layer names.
    """
    result = []
    for name, module in model.named_modules():
        if cond_fn(name, module):
            result.append(name)
    return result


def find_all_linears(model: nn.Module) -> List[str]:
    """Find all linear layers suitable for LoRA in a Megatron model.
    
    
    
    Args:
        model: The Megatron model.
        
    Returns:
        List of layer names suitable for LoRA.
    """
    def _cond(name: str, module: nn.Module) -> bool:
        if name == 'output_layer':
            return False
        if isinstance(module, (TELinear, TELayerNormColumnParallelLinear, TEGroupedLinear, nn.Linear)):
            return True
        return False
    
    return find_layers(model, _cond)


def find_router(model: nn.Module) -> List[str]:
    """Find all MoE router layers in a Megatron model.
    
    
    
    Args:
        model: The Megatron model.
        
    Returns:
        List of router layer names.
    """
    return find_layers(model, lambda name, module: isinstance(module, TopKRouter))


def find_embedding(model: nn.Module) -> List[str]:
    """Find all embedding layers in a Megatron model.
    
    
    
    Args:
        model: The Megatron model.
        
    Returns:
        List of embedding layer names.
    """
    return find_layers(model, lambda name, module: isinstance(module, LanguageModelEmbedding))


def get_target_modules(model: nn.Module, target_modules: List[str]) -> List[str]:
    """Expand target module specifications to actual module names.
    
    
    
    Args:
        model: The Megatron model.
        target_modules: List of target module specs, may include 'all-linear', etc.
        
    Returns:
        Expanded list of target module names.
    """
    result = target_modules.copy()
    if 'all-linear' in result:
        result.remove('all-linear')
        result += find_all_linears(model)
    if 'all-embedding' in result:
        result.remove('all-embedding')
        result += find_embedding(model)
    if 'all-router' in result:
        result.remove('all-router')
        result += find_router(model)
    return list(set(result))


def set_linear_is_expert(model: nn.Module):
    """Mark expert linear layers in MoE models.

    Args:
        model: The Megatron model.
    """
    for name, module in model.named_modules():
        if '.local_experts.' in name and isinstance(
            module, (TELinear, TELayerNormColumnParallelLinear)
        ):
            module.is_expert = True
        elif isinstance(module, TEGroupedLinear):
            module.is_expert = True


def deep_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Get nested attribute using dot notation.
    
    Args:
        obj: The object.
        attr: Dot-separated attribute path.
        default: Default value if attribute not found.
        
    Returns:
        The attribute value or default.
    """
    try:
        for a in attr.split('.'):
            obj = getattr(obj, a)
        return obj
    except AttributeError:
        return default


# =============================================================================

# =============================================================================
def _convert_hf_config(config, _internal_call: bool = False) -> Dict[str, Any]:
    """Convert HuggingFace config to Megatron config dict.
    
    
    
    Args:
        config: HuggingFace model config.
        _internal_call: Internal flag for recursion.
        
    Returns:
        Megatron-compatible config dict.
    """
    megatron_config = {}
    for k, hf_keys in CONFIG_MAPPING.items():
        for hf_k in hf_keys:
            if hasattr(config, hf_k):
                hf_v = getattr(config, hf_k)
                if hf_v is None:
                    continue
                if k == 'rotary_base':
                    megatron_config[k] = int(hf_v)
                elif k in {'untie_embeddings_and_output_weights', 'disable_bias_linear', 'moe_router_pre_softmax'}:
                    megatron_config[k] = not hf_v
                elif k == 'swiglu':
                    if hf_v == 'silu':
                        megatron_config[k] = True
                else:
                    if k == 'kv_lora_rank':
                        megatron_config['multi_latent_attention'] = True
                    elif k == 'architectures':
                        if _internal_call:
                            k = 'llm_architectures'
                    megatron_config[k] = hf_v
                break
                
    # Handle nested configs
    for key in ['text_config', 'llm_config', 'thinker_config']:
        if hasattr(config, key):
            megatron_config.update(_convert_hf_config(getattr(config, key), _internal_call=True))
            
    # Compat llama3 rope scaling
    if getattr(config, 'rope_scaling', None) is not None:
        if isinstance(config.rope_scaling, int):
            megatron_config['rope_scaling'] = {'factor': config.rope_scaling, 'type': 'linear'}
        elif isinstance(config.rope_scaling, dict):
            megatron_config['rope_scaling'] = config.rope_scaling
            
    return megatron_config


def convert_hf_config(config) -> Dict[str, Any]:
    """Convert HuggingFace config to Megatron-compatible config.
    
    
    
    Args:
        config: HuggingFace model config.
        
    Returns:
        Megatron-compatible config dict.
    """
    res = _convert_hf_config(config)
    
    # Process architectures
    architectures = res.get('architectures')
    if isinstance(architectures, list) and architectures:
        architectures = architectures[0]
    res['architectures'] = architectures
    
    llm_architectures = res.get('llm_architectures') or architectures
    if isinstance(llm_architectures, list) and llm_architectures:
        llm_architectures = llm_architectures[0]
    res['llm_architectures'] = llm_architectures
    
    # Process MoE settings
    first_k_dense_replace = res.pop('first_k_dense_replace', None)
    n_shared_experts = res.pop('n_shared_experts', None)
    
    # ==== Qwen2/Qwen2.5 Model specific settings ====
    if llm_architectures == 'Qwen2ForCausalLM':
        # Qwen2/Qwen2.5 uses bias=True for Q, K, V projections (hardcoded in transformers)
        # but the config doesn't have 'attention_bias' field
        if 'add_qkv_bias' not in res:
            res['add_qkv_bias'] = True
        res['swiglu'] = True
        
    # ==== Qwen3 Dense Model specific settings ====
    if llm_architectures == 'Qwen3ForCausalLM':
        res['qk_layernorm'] = True
        # Qwen3 uses SwiGLU activation
        res['swiglu'] = True
        # Qwen3 typically doesn't use bias in linear layers
        res['disable_bias_linear'] = True
        
    # ==== Qwen3 MoE Model specific settings ====
    if llm_architectures == 'Qwen3MoeForCausalLM':
        res['qk_layernorm'] = True
        res['swiglu'] = True
        res['disable_bias_linear'] = True
        # Qwen3 MoE uses shared expert gate
        res['use_shared_expert_gate'] = True
        # Remove ffn_hidden_size as MoE uses moe_ffn_hidden_size
        res.pop('ffn_hidden_size', None)
        
    # DeepSeek models
    if llm_architectures in {'DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM'}:
        res['qk_layernorm'] = True
        res['moe_router_load_balancing_type'] = 'seq_aux_loss'
        res.pop('num_query_groups', None)
        
    # Handle rope scaling
    rope_scaling = res.get('rope_scaling') or {}
    if 'partial_rotary_factor' not in res and 'partial_rotary_factor' in rope_scaling:
        res['partial_rotary_factor'] = rope_scaling['partial_rotary_factor']
    if rope_scaling.get('mrope_section') is not None:
        res['position_embedding_type'] = 'mrope'
        res['mrope_section'] = rope_scaling['mrope_section']
        
    # MoE layer frequency
    if first_k_dense_replace is not None:
        res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if res.get('moe_router_score_function', 'softmax') == 'sigmoid' and 'moe_router_enable_expert_bias' not in res:
        res['moe_router_enable_expert_bias'] = True
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res.get('moe_ffn_hidden_size', res.get('ffn_hidden_size', 0))
        
    return res


@contextmanager
def patch_deepcopy():
    """Context manager to handle tp_group in deepcopy operations.
    
    
    
    WHY THIS IS NECESSARY:
    ----------------------
    Megatron-Core's TransformerEngine linear layers (TELinear, TEColumnParallelLinear, etc.)
    store a reference to their tensor parallel process group in the `tp_group` attribute.
    
    When PEFT's get_peft_model() is called, it internally uses copy.deepcopy() to create
    copies of certain modules. However, torch.distributed.ProcessGroup objects cannot be
    pickled or deepcopied because:
    
    1. ProcessGroup objects contain native CUDA/NCCL handles that are process-specific
    2. These handles cannot be serialized and recreated in a different memory context
    3. Attempting to deepcopy them raises: "RuntimeError: Cannot pickle ProcessGroup"
    
    This patch temporarily sets tp_group to None during deepcopy, then restores it
    after the copy is complete. This allows PEFT to work with Megatron modules while
    preserving the correct process group references.
    
    USAGE:
    ------
    ```python
    with patch_deepcopy():
        model = get_peft_model(megatron_model, lora_config)
    ```
    
    Without this patch, the above code would fail with a pickling error.
    """
    import copy
    _origin_deepcopy = copy.deepcopy

    def new_deepcopy(x, *args, **kwargs):
        if getattr(x, 'tp_group', None) is not None:
            origin_tp_group = x.tp_group
            x.tp_group = None
            res = _origin_deepcopy(x, *args, **kwargs)
            x.tp_group = origin_tp_group
            res.tp_group = origin_tp_group
            return res
        else:
            return _origin_deepcopy(x, *args, **kwargs)

    copy.deepcopy = new_deepcopy
    try:
        yield
    finally:
        copy.deepcopy = _origin_deepcopy


# =============================================================================

# =============================================================================
def tuners_sharded_state_dict(
    module: nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
) -> Dict[str, Any]:
    """Generate sharded state dict for PEFT tuners.
    
    
    
    Args:
        module: The module to generate state dict for.
        prefix: Key prefix.
        sharded_offsets: Sharding offsets for distributed checkpointing.
        metadata: Additional metadata.
        
    Returns:
        Sharded state dictionary.
    """
    sharded_state_dict = {}
    # Save parameters
    module._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
    sharded_state_dict = make_sharded_tensors_for_checkpoint(
        sharded_state_dict, prefix, sharded_offsets=sharded_offsets
    )
    # Recurse into submodules
    for name, child in module.named_children():
        if 'Dict' in child.__class__.__name__:
            modules = child.named_children()
        else:
            modules = [(None, child)]
        for n, m in modules:
            _prefix = f'{prefix}{name}.' if n is None else f'{prefix}{name}.{n}.'
            sharded_state_dict.update(sharded_state_dict_default(m, _prefix, sharded_offsets, metadata))
    return sharded_state_dict


def prepare_mcore_model(
    model: nn.Module,
    train_type: str = 'lora',
    lora_config: Optional[Dict[str, Any]] = None,
    freeze_parameters: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
) -> nn.Module:
    """Prepare Megatron-Core model for training.
    
    Args:
        model: The Megatron model.
        train_type: Training type ('full' or 'lora').
        lora_config: LoRA configuration dict.
        freeze_parameters: List of parameter names to freeze.
        tenant_id: Optional tenant ID for multi-tenant isolation.
        
    Returns:
        Prepared model.
    """
    # Set up tenant context if provided
    context = contextmanager(lambda: (yield))()
    if tenant_id is not None:
        manager = get_tenant_manager()
        context = manager.tenant_context(tenant_id)
        
    with context:
        if train_type == 'full':
            if freeze_parameters:
                for name, param in model.named_parameters():
                    if any(fp in name for fp in freeze_parameters):
                        param.requires_grad = False
        elif train_type == 'lora':
            set_linear_is_expert(model)
            if lora_config is not None:
                model = prepare_lora_model(model, lora_config)
    return model


def prepare_lora_model(
    model: nn.Module,
    lora_config: Dict[str, Any],
) -> nn.Module:
    """Add LoRA adapters to Megatron model.
    
    Args:
        model: The Megatron model.
        lora_config: LoRA configuration dict with keys:
            - r: LoRA rank
            - lora_alpha: LoRA alpha
            - lora_dropout: Dropout rate
            - target_modules: Target module names
            - use_rslora: Use rank-stabilized LoRA
            
    Returns:
        Model with LoRA adapters.
    """
    set_linear_is_expert(model)
    
    target_modules = get_target_modules(model, lora_config.get('target_modules', ['all-linear']))
    
    peft_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.0),
        target_modules=target_modules,
        bias=lora_config.get('bias', 'none'),
        use_rslora=lora_config.get('use_rslora', False),
    )
    
    with patch_deepcopy():
        model = get_peft_model(model, peft_config)
        
    return model


# =============================================================================

# =============================================================================
def get_local_layer_specs(config, layer_specs: List, vp_stage: Optional[int] = None):
    """Get local layer specifications for current pipeline rank.
    
    
    
    Args:
        config: Megatron transformer config.
        layer_specs: Full list of layer specifications.
        vp_stage: Virtual pipeline stage index.
        
    Returns:
        Local layer specifications for this rank.
    """
    kwargs = {'vp_stage': vp_stage} if mcore_013 else {}
    num_layers_to_build = get_num_layers_to_build(config, **kwargs)

    if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, **kwargs)
        ]
    else:
        offset = get_transformer_layer_offset(config, **kwargs)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    return local_layer_specs


def get_padding_to(
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    fp8_format: Optional[str] = None,
    fp8_recipe: Optional[str] = None,
    attention_backend: Optional[str] = None,
) -> Optional[int]:
    """Get padding size for sequence length.
    
    Args:
        tensor_model_parallel_size: TP size.
        context_parallel_size: CP size.
        sequence_parallel: Whether sequence parallel is enabled.
        fp8_format: FP8 format if used.
        fp8_recipe: FP8 recipe if used.
        attention_backend: Attention backend type.
        
    Returns:
        Padding size or None.
    """
    padding_to = None
    if tensor_model_parallel_size > 1 and sequence_parallel:
        padding_to = tensor_model_parallel_size
    if context_parallel_size > 1:
        padding_to = (padding_to or 1) * context_parallel_size
    origin_padding_to = padding_to
    
    if fp8_recipe == 'blockwise':
        padding_to = (padding_to or 1) * 128
    elif fp8_format is not None:
        padding_to = max((padding_to or 1) * 8, 16)
        
    if attention_backend == 'fused':
        padding_to = max(padding_to or 1, ((origin_padding_to) or 1) * 64)
        
    return padding_to


# =============================================================================

# =============================================================================
def forward_step_helper(model: nn.Module, inputs: Dict[str, Any], config) -> Optional[torch.Tensor]:
    """Helper for pipeline parallel forward step.
    
    Handles communication between pipeline stages.
    
    Args:
        model: The model.
        inputs: Input dict with position_ids, etc.
        config: Configuration with parallel settings.
        
    Returns:
        Output tensor for last stage, None otherwise.
    """
    from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
    
    if mpu.is_pipeline_first_stage():
        # Get micro_batch_size from input tensor, not config
        # For padding_free (qkv_format 'thd'), use 1
        micro_batch_size = 1
        if not getattr(config, 'padding_free', False):
            # Infer batch size from input_ids or position_ids
            if 'input_ids' in inputs:
                micro_batch_size = inputs['input_ids'].shape[0]
            elif 'position_ids' in inputs:
                micro_batch_size = inputs['position_ids'].shape[0]
            else:
                micro_batch_size = 1
        seq_length = inputs['position_ids'].shape[-1]
        if config.sequence_parallel:
            seq_length //= mpu.get_tensor_model_parallel_world_size()
        recv_shape_buffer = torch.tensor(
            [seq_length, micro_batch_size, config.hidden_size],
            device=Platform.get_local_device(),
            dtype=torch.int64
        )
    else:
        recv_shape_buffer = torch.empty((3,), device=Platform.get_local_device(), dtype=torch.int64)
        recv_from_prev_pipeline_rank_(recv_shape_buffer)
        
    if not mpu.is_pipeline_last_stage():
        send_to_next_pipeline_rank(recv_shape_buffer)
    shape = recv_shape_buffer.tolist()

    if not mpu.is_pipeline_first_stage():
        dtype = config.params_dtype
        recv_buffer = torch.empty(shape, device=Platform.get_local_device(), dtype=dtype)
        recv_from_prev_pipeline_rank_(recv_buffer)
        model.set_input_tensor(recv_buffer)
        
    output_tensor = model(**inputs)
    
    if not mpu.is_pipeline_last_stage():
        send_to_next_pipeline_rank(output_tensor)
        output_tensor = None

    return output_tensor


class MegatronTrainerState:
    """Lightweight trainer state for Megatron training.
    
    Provides compatibility with transformers TrainerState interface.
    
    Attributes:
        global_step: The current training step.
        max_steps: The total number of training steps.
    """

    def __init__(self, global_step: int = 0, max_steps: int = 0):
        self.global_step = global_step
        self.max_steps = max_steps

    def update(self, global_step: Optional[int] = None, max_steps: Optional[int] = None):
        if global_step is not None:
            self.global_step = global_step
        if max_steps is not None:
            self.max_steps = max_steps

    def __repr__(self) -> str:
        return f'MegatronTrainerState(global_step={self.global_step}, max_steps={self.max_steps})'


def get_model_parameter_info(model: nn.Module) -> Dict[str, Any]:
    """Get parameter count information for a model.
    
    Args:
        model: The model.
        
    Returns:
        Dict with total_params, trainable_params, frozen_params.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
    }
