# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor


def _tiny_qwen3_dir():
    path = tempfile.mkdtemp()
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    return path


def _sample(length, generator):
    input_ids = torch.randint(0, 64, (length,), generator=generator)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(length),
    }


def test_forward_only_keeps_padding_free_samples_apart():
    """A sample packed after another must get the same log-probs as on its own."""
    model = TransformersModel(model_id=_tiny_qwen3_dir(), mixed_precision="no")
    model.set_processor(InputProcessor, padding_free=True)
    model._lazy_wrap_model()
    device = str(next(model.model.parameters()).device)
    generator = torch.Generator().manual_seed(0)
    first, second = _sample(12, generator), _sample(20, generator)
    with patch("twinkle.processor.base.Platform.get_local_device", return_value=device):
        packed = model.forward_only(inputs=[first, second])["logps"]
        alone = model.forward_only(inputs=[second])["logps"]
    torch.testing.assert_close(packed[1], alone[0])

def test_forward_keeps_padding_free_samples_apart_without_checkpointing():
    """Without gradient checkpointing the training forward must not build a KV cache either."""
    model = TransformersModel(model_id=_tiny_qwen3_dir(), mixed_precision="no", gradient_checkpointing=False)
    model.set_processor(InputProcessor, padding_free=True)
    model._lazy_wrap_model()
    model.model.train()
    device = str(next(model.model.parameters()).device)
    generator = torch.Generator().manual_seed(0)
    first, second = _sample(12, generator), _sample(20, generator)
    with patch("twinkle.processor.base.Platform.get_local_device", return_value=device):
        packed = model.forward(inputs=[first, second])["logps"]
        alone = model.forward(inputs=[second])["logps"]
    torch.testing.assert_close(packed[1].detach(), alone[0].detach())
