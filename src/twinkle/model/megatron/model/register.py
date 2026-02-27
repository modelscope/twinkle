# Copyright (c) ModelScope Contributors. All rights reserved.
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Type

from .constant import MLLMMegatronModelType

MEGATRON_MODEL_MAPPING = {}


@dataclass
class MegatronModelMeta:
    megatron_model_type: str
    model_types: List[str]

    is_multimodal: bool = False
    bridge_cls: Optional[Type] = None
    model_cls: Optional[Type[nn.Module]] = None
    visual_cls: Optional[Type[nn.Module]] = None
    auto_model_cls: Optional[Type] = None
    loader: Optional[Type['MegatronModelLoader']] = None

    def __post_init__(self):
        if self.megatron_model_type in MLLMMegatronModelType.__dict__:
            self.is_multimodal = True
        if self.bridge_cls is None:
            from .gpt_bridge import GPTBridge, MultimodalGPTBridge
            self.bridge_cls = MultimodalGPTBridge if self.is_multimodal else GPTBridge
        if self.model_cls is None:
            from .gpt_model import GPTModel
            from .mm_gpt_model import MultimodalGPTModel
            self.model_cls = MultimodalGPTModel if self.is_multimodal else GPTModel
        if self.auto_model_cls is None:
            from transformers import AutoModel, AutoModelForCausalLM
            self.auto_model_cls = AutoModel if self.is_multimodal else AutoModelForCausalLM
        if self.loader is None:
            self.loader = MegatronModelLoader


class MegatronModelLoader:
    """Default loader that builds TransformerConfig + layer specs for a model.

    Subclass this to customize layer spec construction (e.g. heterogeneous
    attention types, custom layer norms). Register the subclass via
    ``MegatronModelMeta(loader=MyLoader)``.
    """

    def get_layer_spec(self, config, args, mg_config_dict):
        """Build a transformer layer spec from *config* (``TransformerConfig``).

        The default implementation delegates to Megatron-Core's
        ``get_gpt_layer_with_transformer_engine_spec``.

        Returns:
            A ``ModuleSpec`` or ``TransformerBlockSubmodules`` instance.
        """
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
        num_experts = mg_config_dict.get('num_experts', 0) or 0
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=num_experts > 0,
            qk_layernorm=mg_config_dict.get('qk_layernorm', False),
        )

    def get_mtp_block_spec(self, config, layer_spec, **kwargs):
        """Build MTP block spec. Override for custom layer norms etc."""
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
        return get_gpt_mtp_block_spec(config, layer_spec, use_transformer_engine=True, **kwargs)

    def post_config(self, config, args, mg_config_dict):
        """Hook called after TransformerConfig is created but before layer specs.

        Use this to set model-specific config attributes (e.g. ``layer_types``,
        ``moe_use_shared_expert_gate``).
        """
        pass


def register_megatron_model(megatron_model_meta: MegatronModelMeta, *, exist_ok: bool = False):
    megatron_model_type = megatron_model_meta.megatron_model_type
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
