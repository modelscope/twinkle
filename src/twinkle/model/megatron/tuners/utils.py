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
# Import lora module to register dispatch_megatron with peft
from . import lora as _ 

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


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
