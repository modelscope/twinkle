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
