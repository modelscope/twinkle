# Copyright (c) ModelScope Contributors. All rights reserved.
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, List, Optional, Type

import torch.nn as nn

from .constant import MLLMMegatronModelType

MEGATRON_MODEL_MAPPING = {}


@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]

    is_multimodal: bool = False
    bridge_cls: Optional[Type] = None
    model_cls: Optional[Type[nn.Module]] = None
    get_transformer_layer_spec: Optional[Callable] = None
    model_provider: Optional[Callable[[], nn.Module]] = None
    visual_cls: Optional[Type[nn.Module]] = None
    get_mtp_block_spec: Optional[Callable] = None
    # AutoModel class for loading HF model (AutoModelForCausalLM for text, AutoModel for multimodal)
    auto_model_cls: Optional[Type] = None

    extra_args_provider: Optional[Callable[[ArgumentParser], ArgumentParser]] = None

    def __post_init__(self):
        if self.megatron_model_type in MLLMMegatronModelType.__dict__:
            self.is_multimodal = True
        if self.bridge_cls is None:
            from .gpt_bridge import GPTBridge, MultimodalGPTBridge
            self.bridge_cls = MultimodalGPTBridge if self.is_multimodal else GPTBridge
        if self.model_cls is None:
            from .mm_gpt_model import MultimodalGPTModel
            from megatron.core.models.gpt import GPTModel
            self.model_cls = MultimodalGPTModel if self.is_multimodal else GPTModel
        if self.auto_model_cls is None:
            from transformers import AutoModelForCausalLM, AutoModel
            self.auto_model_cls = AutoModel if self.is_multimodal else AutoModelForCausalLM


def register_megatron_model(megatron_model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = megatron_model_meta.megatron_model_type
    # diff here
    if not exist_ok and megatron_model_type in MEGATRON_MODEL_MAPPING:
        raise ValueError(f'The `{megatron_model_type}` has already been registered in the MEGATRON_MODEL_MAPPING.')
    MEGATRON_MODEL_MAPPING[megatron_model_type] = megatron_model_meta


_MODEL_META_MAPPING = None


def get_megatron_model_meta(model_type: str) -> Optional[MegatronModelMeta]:
    global _MODEL_META_MAPPING
    if _MODEL_META_MAPPING is None:
        _MODEL_META_MAPPING = {}
        for k, megatron_model_meta in MEGATRON_MODEL_MAPPING.items():
            for _model_type in megatron_model_meta.model_types:
                _MODEL_META_MAPPING[_model_type] = k
    if model_type not in _MODEL_META_MAPPING:
        return
    return MEGATRON_MODEL_MAPPING[_MODEL_META_MAPPING[model_type]]
