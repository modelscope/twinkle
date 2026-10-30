# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise dev task entry points on real tiny decoder models without downloads."""
import copy

import pytest
import torch


@pytest.mark.parametrize('task', ['embedding', 'reranker', 'seq_cls'])
def test_dev_task_forward_backward_and_update(tmp_path, monkeypatch, task):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                              Qwen2ForSequenceClassification)
    from twinkle import Platform, initialize
    from twinkle.model.transformers.transformers import TransformersModel
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, pad_token_id=0,
                         num_labels=3 if task == 'seq_cls' else 1)
    model_cls = Qwen2ForCausalLM if task == 'embedding' else Qwen2ForSequenceClassification
    model_cls(config).save_pretrained(tmp_path)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    vocab.update({f't{i}': i for i in range(5, 32)})
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordPiece(vocab, unk_token='[UNK]')),
                           pad_token='[PAD]', unk_token='[UNK]').save_pretrained(tmp_path)
    model = TransformersModel(model_id=str(tmp_path), mixed_precision='no', gradient_checkpointing=False)
    model.set_processor('InputProcessor', padding_side='right')
    if task == 'embedding':
        model.set_loss('InfonceLoss', temperature=0.1)
        labels = [1, 0, 0, 1, 0, 0]
    elif task == 'reranker':
        model.set_loss('PointwiseRerankerLoss')
        labels = [1., 0., 0., 1., 0., 0.]
    else:
        model.set_loss('SeqClsLoss', problem_type='single_label_classification', num_labels=3)
        labels = [0, 1, 2, 0, 1, 2]
    model.set_optimizer('AdamW', lr=1e-3)
    inputs = [{'input_ids': [2, 5+i, 3], 'attention_mask': [1] * 3, 'labels': [label]}
              for i, label in enumerate(labels)]
    before = {name: p.detach().clone() for name, p in model.model.named_parameters()}
    result = model.forward_backward(inputs=copy.deepcopy(inputs), task=task)
    assert torch.isfinite(torch.tensor(result['loss']))
    model.clip_grad_and_step()
    assert any(not torch.equal(before[name], p) for name, p in model.model.named_parameters())
    result = model.forward_only(inputs=copy.deepcopy(inputs), task=task, return_logits=True)
    output = result['embeddings' if task == 'embedding' else 'logits']
    assert output.shape[0] == len(inputs) and torch.isfinite(output).all()


@pytest.mark.parametrize('task', ['embedding', 'reranker', 'seq_cls'])
def test_head_lora_tenants_and_cold_optimizer_resume(tmp_path, monkeypatch, task):
    from peft import LoraConfig
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from cookbook.client.twinkle.prepare_sequence_model import prepare_models, tiny_backbone
    from cookbook.client.twinkle import embedding, reranker, seq_cls
    from twinkle import Platform, initialize
    from twinkle.model.transformers.multi_lora_transformers import MultiLoraTransformersModel

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    paths = prepare_models(tiny_backbone(tmp_path / 'source'), tmp_path / 'bases')
    targets = ['q_proj', 'v_proj'] + ([] if task == 'embedding' else ['score'])
    recipe = {'embedding': embedding, 'reranker': reranker, 'seq_cls': seq_cls}[task]
    batch = recipe.encoded_batch(AutoTokenizer.from_pretrained(paths[task]))
    assert all(row['input_ids'] for row in batch), batch

    def build():
        return MultiLoraTransformersModel(model_id=paths[task], mixed_precision='no',
                                         gradient_checkpointing=False, max_loras=2, max_r=8,
                                         target_modules=targets)

    def add(model, name):
        model.add_adapter_to_model(name, LoraConfig(r=4, lora_alpha=8, target_modules=targets),
                                   gradient_accumulation_steps=2)
        model.set_processor('InputProcessor', padding_side='right', adapter_name=name)
        if task == 'embedding':
            model.set_loss('InfonceLoss', temperature=0.1, use_batch=True, adapter_name=name)
        elif task == 'reranker':
            model.set_loss('PointwiseRerankerLoss', adapter_name=name)
        else:
            model.set_loss('SeqClsLoss', problem_type='single_label_classification',
                           num_labels=3, adapter_name=name)
        model.set_optimizer('Adam', lr=1e-3, adapter_name=name)

    def read(model, name):
        return model.forward_only(inputs=copy.deepcopy(batch), task=task,
                                  return_logits=True, adapter_name=name)[
                                      'embeddings' if task == 'embedding' else 'logits'].detach().clone()

    def step(model, name):
        model.forward_backward(inputs=copy.deepcopy(batch), task=task, adapter_name=name)
        model.clip_grad_and_step(adapter_name=name)

    model = build()
    add(model, 'a')
    add(model, 'b')
    base = {k: p.detach().clone() for k, p in model.model.named_parameters() if 'lora_' not in k}
    initial, peer = read(model, 'a'), read(model, 'b')
    for _ in range(3):
        step(model, 'a')
    trained = read(model, 'a')
    assert not torch.equal(initial, trained)
    torch.testing.assert_close(read(model, 'b'), peer, rtol=0, atol=0)
    for key, parameter in model.model.named_parameters():
        if key in base:
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, base[key], rtol=0, atol=0)
    saved = model.save('trained', output_dir=str(tmp_path), adapter_name='a',
                       save_optimizer=True, consumed_train_samples=3 * len(batch))
    if task != 'embedding':
        head = {k: v for k, v in load_file(saved + '/adapter_model.safetensors').items() if 'score.' in k}
        assert head and any(torch.count_nonzero(v) for k, v in head.items() if 'lora_B' in k)
    step(model, 'a')
    step(model, 'a')
    continued = read(model, 'a')
    model.remove_adapter('a')
    add(model, 'c')
    assert not model.optimizer_group['c'].optimizer.state
    assert model.optimizer_group['c'].cur_step == 0
    torch.testing.assert_close(read(model, 'c'), initial, rtol=0, atol=0)
    model.resume_from_checkpoint(saved, adapter_name='c')
    step(model, 'c')
    step(model, 'c')
    torch.testing.assert_close(read(model, 'c'), continued, rtol=0, atol=1e-6)
    cold = build()
    add(cold, 'a')
    progress = cold.resume_from_checkpoint(saved, adapter_name='a')
    assert progress['cur_step'] == 3
    step(cold, 'a')
    step(cold, 'a')
    torch.testing.assert_close(read(cold, 'a'), continued, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match='modules_to_save'):
        cold.add_adapter_to_model('bad', LoraConfig(target_modules=targets, modules_to_save=['score']))
