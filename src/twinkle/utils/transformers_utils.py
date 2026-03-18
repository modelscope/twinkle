# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import TYPE_CHECKING, Callable, List, Optional

from .utils import deep_getattr

if TYPE_CHECKING:
    import torch.nn as nn


def find_layers(
    model: 'nn.Module',
    cond: Callable[[str, 'nn.Module'], bool],
    sub_module: Optional[str] = None,
    min_name_len: Optional[int] = None,
) -> List[str]:
    # The content of target_module_names cannot exist in inner_nodes.
    sub_module_str = sub_module
    if sub_module is None:
        sub_module = model
    else:
        sub_module = deep_getattr(model, sub_module)
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r'\d+\.', '{}.', name)
        if not cond(name, module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in sub_module.named_modules():
        if sub_module_str:
            name = f'{sub_module_str}.{name}' if name else sub_module_str
        if cond(name, module):
            module_name_list = name.split('.')
            module_name = module_name_list.pop()
            i = 1
            for inner_node in inner_nodes:
                while module_name_list and inner_node.endswith(re.sub(
                        r'\d+\.', '{}.', module_name)) or min_name_len and i < min_name_len:
                    module_name = f'{module_name_list.pop()}.{module_name}'
                    i += 1
            target_module_names.add(module_name)
    return list(target_module_names)


def find_all_linears(model, model_arch=None, extra_layers=None, sub_module=None):
    if model_arch is None:
        model_arch = model.model_meta.model_arch
    # lm_head
    if model_arch and model_arch.lm_head:
        output = model_arch.lm_head
        idx = output.rfind('.')
        lm_head_name = output[idx + 1:]
    else:
        lm_head_name = 'lm_head'
    # 'score', 'classifier': classification model
    # 'v_head': reward model
    ignore_layers = [lm_head_name, 'score', 'v_head', 'classifier'] + ['lora_A', 'lora_B', 'base_layer']
    ignore_linear_cls = [
        'glulinear'  # phi4-mm
    ]

    def _cond(name, module):
        module_name = module.__class__.__name__.lower()
        if (extra_layers and isinstance(module, tuple(extra_layers)) or
            ('linear' in module_name and all(linear_cls not in module_name
                                             for linear_cls in ignore_linear_cls))) and all(layer not in name
                                                                                            for layer in ignore_layers):
            return True
        return False

    return find_layers(model, _cond, sub_module=sub_module)


def get_modules_to_not_convert(model):
    if not hasattr(model, 'model_meta') or not hasattr(model, 'model_info'):
        return
    model_arch = model.model_meta.model_arch
    prefix_list = []
    suffix_list = []
    if model.model_info.is_moe_model:
        suffix_list += ['mlp.gate', 'mlp.shared_expert_gate']
    if model_arch is not None:
        for key in ['vision_tower', 'aligner']:
            value = getattr(model_arch, key, None)
            if value:
                prefix_list += value
    suffix_list.append('lm_head')
    res = []
    for n, m in model.named_modules():
        if 'linear' in m.__class__.__name__.lower() and (any(n.endswith(suffix) for suffix in suffix_list)
                                                         or any(n.startswith(prefix) for prefix in prefix_list)):
            res.append(n)
    return res if res else None


def get_llm_model(model, *, model_meta=None, inner_backbone: bool = True):
    """Best-effort extraction of the LLM module from a (possibly wrapped) model.

    This mirrors the common pattern used by Swift/PEFT/Accelerate stacks:
    - unwrap parallel wrappers (DDP/FSDP/Accelerate)
    - unwrap PEFT/Swift wrappers (if present)
    - use `model_meta.model_arch.language_model` to locate the LLM in multimodal models
    - optionally return the inner backbone (e.g. `QwenModel`/`LlamaModel`) via `.model`
    """
    # 1) Unwrap parallel wrappers (Accelerate).
    try:
        from accelerate.utils import extract_model_from_parallel  # type: ignore

        model = extract_model_from_parallel(model)
    except Exception:
        pass

    # 2) Unwrap PEFT wrappers.
    try:
        from peft import PeftModel  # type: ignore

        if isinstance(model, PeftModel):
            model = model.model
    except Exception:
        pass

    # 3) Locate the language model module in multimodal containers via model_meta.
    if model_meta is None:
        model_meta = getattr(model, 'model_meta', None)
    llm_model = model
    model_arch = getattr(model_meta, 'model_arch', None) if model_meta is not None else None
    llm_prefix = getattr(model_arch, 'language_model', None) if model_arch is not None else None
    if llm_prefix:
        # Convention: `language_model` is a list of candidate prefixes.
        llm_model = deep_getattr(model, llm_prefix[0])
    else:
        llm_model = getattr(model, 'language_model', model)

    # 4) Return the inner backbone if requested.
    if inner_backbone:
        if hasattr(llm_model, 'thinker'):
            llm_model = llm_model.thinker.model
        elif hasattr(llm_model, 'model'):
            llm_model = llm_model.model
    return llm_model
