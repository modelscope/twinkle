import inspect
import tempfile
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from twinkle.loss import PPOValueLoss
from twinkle.model import TransformersValueModel


def _tiny_model_dir():
    path = tempfile.mkdtemp()
    config = GPT2Config(
        vocab_size=8,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    GPT2LMHeadModel(config).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({
        '[UNK]': 0,
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
    }, unk_token='[UNK]'))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[UNK]').save_pretrained(path)
    return path


def test_value_model_constructor_exposes_device_mesh():
    """Keep ``device_mesh`` visible to ``@remote_class`` dispatch."""
    parameter = inspect.signature(TransformersValueModel.__init__).parameters['device_mesh']
    assert parameter.default is None


def test_value_model_forward_and_backward():
    model = TransformersValueModel(model_id=_tiny_model_dir(), mixed_precision='no')
    model.set_loss(PPOValueLoss())
    model._lazy_wrap_model()
    model_device = str(next(model.model.parameters()).device)
    with patch('twinkle.processor.base.Platform.get_local_device', return_value=model_device):
        outputs = model.forward_backward(
            inputs=[{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}],
            old_values=[[0.0, 0.0]],
            returns=[[1.0, 1.0]],
        )
    assert outputs['values'].shape == (1, 3)
    assert outputs.get('logps') is None
    head = model.model.get_output_embeddings()
    assert head.out_features == 1
    assert head.weight.grad is not None
    assert torch.isfinite(head.weight.grad).all()


def test_value_model_forward_only_returns_values():
    model = TransformersValueModel(model_id=_tiny_model_dir(), mixed_precision='no')
    model.set_loss(PPOValueLoss())
    model._lazy_wrap_model()
    model_device = str(next(model.model.parameters()).device)
    with patch('twinkle.processor.base.Platform.get_local_device', return_value=model_device):
        outputs = model.forward_only(
            inputs=[{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}],
        )
    assert outputs['values'].shape == (1, 3)
    assert outputs.get('logps') is None
    assert not outputs['values'].requires_grad


def test_freeze_attention_keeps_mlp_and_value_head_trainable():
    model = TransformersValueModel(model_id=_tiny_model_dir(), mixed_precision='no')
    summary = model.freeze_attention_for_value_training()
    assert summary['attention_modules'] == 1
    backbone = model.model.transformer.h[0]
    assert all(not parameter.requires_grad for parameter in backbone.attn.parameters())
    assert any(parameter.requires_grad for parameter in backbone.mlp.parameters())
    assert all(parameter.requires_grad for parameter in model.model.get_output_embeddings().parameters())
    counts = model.trainable_parameter_summary()
    assert counts['frozen_parameters'] == summary['frozen_parameters']


def test_freeze_attention_supports_qwen35_hybrid_token_mixers():
    """Qwen3.5 exposes full and linear attention under different names."""

    class FullAttentionLayer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Linear(8, 8)
            self.mlp = torch.nn.Linear(8, 8)

    class LinearAttentionLayer(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear_attn = torch.nn.Linear(8, 8)
            self.mlp = torch.nn.Linear(8, 8)

    model = TransformersValueModel(model_id=_tiny_model_dir(), mixed_precision='no')
    full_attention_layer = FullAttentionLayer()
    linear_attention_layer = LinearAttentionLayer()
    model.model.qwen35_hybrid_layers = torch.nn.ModuleList([
        full_attention_layer,
        linear_attention_layer,
    ])

    summary = model.freeze_attention_for_value_training()

    # One GPT-2 ``attn`` module plus the two Qwen3.5-style token mixers.
    assert summary['attention_modules'] == 3
    assert all(not parameter.requires_grad for parameter in full_attention_layer.self_attn.parameters())
    assert all(not parameter.requires_grad for parameter in linear_attention_layer.linear_attn.parameters())
    assert all(parameter.requires_grad for parameter in full_attention_layer.mlp.parameters())
    assert all(parameter.requires_grad for parameter in linear_attention_layer.mlp.parameters())
    assert all(parameter.requires_grad for parameter in model.model.get_output_embeddings().parameters())
