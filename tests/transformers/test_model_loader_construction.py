# Copyright (c) ModelScope Contributors. All rights reserved.
"""A supplied loader owns external architectures, including empty-weight construction."""
import pytest
import torch


@pytest.mark.parametrize('multi', [False, True])
def test_loader_builds_architecture_outside_transformers(tmp_path, monkeypatch, multi):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from twinkle import Platform, initialize
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel
    from twinkle.model.transformers.transformers import TransformersModel

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0)
    config.architectures = ['ExternalArchitectureOwnedByLoader']
    config.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordPiece({'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]')),
        pad_token='[PAD]', unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path)
    calls = []

    class Loader:
        def build_config(self, path):
            return config

        def process_config(self, value):
            return value

        def build_processor(self, path, value):
            calls.append('processor')
            return tokenizer

        def process_tokenizer(self, value):
            return value

        def build_model(self, path, **kwargs):
            calls.append('model')
            return Qwen2ForCausalLM(kwargs['config'])

        def build_model_from_config(self, value, **kwargs):
            calls.append('meta')
            return Qwen2ForCausalLM(value)

        def process_model(self, value):
            return value

    loader = Loader()
    cls = MultiLoraTransformersModel if multi else TransformersModel
    wrapper = cls(model_id=str(tmp_path), model_loader=loader, mixed_precision='no',
                  gradient_checkpointing=False, **({'max_loras': 2, 'max_r': 4,
                                                   'target_modules': ['q_proj']} if multi else {}))
    assert calls == ['processor', 'model']
    empty = wrapper._init_empty_model_from_config(None, model_loader=loader, processor=tokenizer)
    assert all(p.device.type == 'meta' for p in empty.parameters())
    assert calls[-1] == 'meta'
