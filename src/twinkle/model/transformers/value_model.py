# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from torch import nn
from typing import Optional

from twinkle import DeviceMesh, remote_class
from .transformers import TransformersModel


@remote_class()
class TransformersValueModel(TransformersModel):
    """Transformers causal-LM backbone with a scalar value head."""

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
