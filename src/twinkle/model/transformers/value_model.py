# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from torch import nn
from typing import Optional

from twinkle import DeviceMesh, remote_class, remote_function
from .transformers import TransformersModel


@remote_class()
class TransformersValueModel(TransformersModel):
    """Transformers causal-LM backbone with a scalar value head.

    The implementation supports causal language models with hybrid attention.
    In particular, Qwen3.5 alternates full-attention decoder layers
    (``self_attn``) with GatedDeltaNet linear-attention layers
    (``linear_attn``).  Both token mixers are treated as attention modules
    when applying the frozen-attention critic training strategy.
    """

    _ATTENTION_ATTRIBUTES = ('self_attn', 'linear_attn', 'attn')

    def __init__(self, *args, device_mesh: Optional[DeviceMesh] = None, **kwargs):
        super().__init__(*args, device_mesh=device_mesh, **kwargs)
        output_head = self.model.get_output_embeddings()
        if output_head is None or not hasattr(output_head, 'in_features'):
            raise ValueError('The model must expose a linear output embedding to construct a value head')
        value_head = nn.Linear(
            output_head.in_features, 1, bias=True, device=output_head.weight.device, dtype=output_head.weight.dtype)
        nn.init.zeros_(value_head.weight)
        nn.init.zeros_(value_head.bias)
        self.model.set_output_embeddings(value_head)
        self.model.config.tie_word_embeddings = False

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def freeze_attention_for_value_training(self):
        """Freeze attention/token-mixer modules while leaving feed-forward/value-head parameters trainable.

        ``self_attn`` covers conventional and Qwen3.5 full-attention decoder
        layers, ``linear_attn`` covers Qwen3.5 GatedDeltaNet layers, and
        ``attn`` preserves compatibility with models
        such as GPT-2 and vision backbones.
        """
        model = self.strategy.unwrap_model(self.model)
        attention_modules = []
        seen_attention_ids = set()
        for module in model.modules():
            for attribute in self._ATTENTION_ATTRIBUTES:
                attention = getattr(module, attribute, None)
                if isinstance(attention, nn.Module) and id(attention) not in seen_attention_ids:
                    attention_modules.append(attention)
                    seen_attention_ids.add(id(attention))
        if not attention_modules:
            raise ValueError(f'No decoder attention modules found for {type(model).__name__}')
        frozen = 0
        for attention in attention_modules:
            for parameter in attention.parameters():
                parameter.requires_grad = False
                frozen += parameter.numel()
        return {'attention_modules': len(attention_modules), 'frozen_parameters': frozen}

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def trainable_parameter_summary(self):
        model = self.strategy.unwrap_model(self.model)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return {'total_parameters': total, 'trainable_parameters': trainable, 'frozen_parameters': total - trainable}

    def add_adapter_to_model(self, *args, **kwargs):
        raise NotImplementedError('PPO critic LoRA is not supported; train the critic as a full-parameter model')

    def forward(self, *, inputs, **kwargs):
        task = kwargs.get('task', 'causal_lm')
        if task != 'causal_lm':
            raise ValueError('TransformersValueModel only supports task="causal_lm"')
        return super().forward(inputs=inputs, **kwargs)

    def forward_only(self, *, inputs, **kwargs):
        task = kwargs.get('task', 'causal_lm')
        if task != 'causal_lm':
            raise ValueError('TransformersValueModel only supports task="causal_lm"')
        return super().forward_only(inputs=inputs, **kwargs)
