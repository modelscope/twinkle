# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real offline ST pipeline: tenant isolation, optimizer continuation and slot reuse."""
import copy

import pytest
import torch
from peft import LoraConfig, PeftModel

st = pytest.importorskip('sentence_transformers')
pytest.importorskip('swift.dev.model.multi_lora_sentence_transformer_model')


def test_st_pipeline_tenants_checkpoint_and_reuse(tmp_path, monkeypatch):
    from sentence_transformers.base.modules import Transformer, Dense, Normalize
    from sentence_transformers.sentence_transformer.modules import Pooling
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from swift.dev.model.multi_lora_sentence_transformer_model import MultiLoraSentenceTransformerModel
    from twinkle import Platform, initialize

    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: 'cpu')
    initialize(mode='local')
    torch.manual_seed(42)
    backbone = tmp_path / 'bert'
    BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0)).save_pretrained(backbone)
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
    vocab.update({f't{i}': i for i in range(5, 32)})
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordPiece(vocab, unk_token='[UNK]')),
                           pad_token='[PAD]', unk_token='[UNK]').save_pretrained(backbone)
    original = tmp_path / 'st'
    st.SentenceTransformer(modules=[Transformer(str(backbone)), Pooling(16),
                                    Dense(16, 8), Normalize()], device='cpu').save(str(original))
    module_config = (original / 'modules.json').read_bytes()
    model = MultiLoraSentenceTransformerModel(model_id=str(original), mixed_precision='no',
                                             gradient_checkpointing=False, max_loras=2, max_r=4,
                                             target_modules=['query', 'value'])
    batch = [{'input_ids': [2, 5 + i, 3], 'attention_mask': [1] * 3, 'labels': [int(i % 3 == 0)]}
             for i in range(6)]

    def add(name, lr):
        model.add_adapter_to_model(name, LoraConfig(r=4, target_modules=['query', 'value']))
        model.set_processor('InputProcessor', padding_side='right', adapter_name=name)
        model.set_loss('InfonceLoss', temperature=0.1, adapter_name=name)
        model.set_optimizer('AdamW', lr=lr, adapter_name=name)

    def state(name):
        return {k: v.detach().cpu().clone() for k, v in model.get_state_dict(adapter_name=name).items()}

    def vectors(name, **kwargs):
        return model.forward_only(inputs=copy.deepcopy(batch), adapter_name=name, **kwargs)['embeddings'].cpu()

    def step(name):
        model.forward_backward(inputs=copy.deepcopy(batch), adapter_name=name, task='embedding')
        model.clip_grad_and_step(adapter_name=name)

    add('a', 1e-3)
    add('b', 2e-3)
    group_a, group_b = model.optimizer_group['a'], model.optimizer_group['b']
    template_a, template_b = group_a.template, group_b.template
    assert template_a is not template_b
    model.set_template('Template', max_length=17, adapter_name='b')
    assert group_a.template is template_a and group_b.template is not template_b
    template_b = group_b.template
    a0, b0, b_vectors = state('a'), state('b'), vectors('b')
    reference = st.SentenceTransformer(str(original), device='cpu').eval()
    features = {'input_ids': torch.tensor([row['input_ids'] for row in batch]),
                'attention_mask': torch.ones(6, 3, dtype=torch.long)}
    with torch.no_grad():
        raw = reference(dict(features))['sentence_embedding']
    torch.testing.assert_close(vectors('a', disable_lora=True), raw, rtol=0, atol=1e-6)
    step('a')
    assert any(not torch.equal(a0[k], v) for k, v in state('a').items())
    for key, value in state('b').items():
        torch.testing.assert_close(value, b0[key], rtol=0, atol=0)
    torch.testing.assert_close(vectors('b'), b_vectors, rtol=0, atol=0)
    assert not group_b.optimizer.state and group_b.cur_step == 0
    assert group_a.template is template_a and group_b.template is template_b
    saved_vectors = vectors('a')
    checkpoint = model.save('a', output_dir=str(tmp_path), adapter_name='a', save_optimizer=True,
                            consumed_train_samples=6)
    step('a')
    continued = vectors('a')
    model.remove_adapter('a')
    add('c', 1e-3)
    group_c = model.optimizer_group['c']
    assert not group_c.optimizer.state and group_c.cur_step == 0
    assert group_c.template is not template_a and group_c.template is not template_b
    for key, value in state('c').items():
        torch.testing.assert_close(value, a0[key], rtol=0, atol=0)
    progress = model.resume_from_checkpoint(checkpoint, adapter_name='c')
    assert progress['consumed_train_samples'] == 6 and progress['cur_step'] == 1
    assert model.optimizer_group['c'].optimizer.state
    torch.testing.assert_close(vectors('c'), saved_vectors, rtol=0, atol=1e-6)
    step('c')
    torch.testing.assert_close(vectors('c'), continued, rtol=0, atol=1e-6)
    torch.testing.assert_close(vectors('b'), b_vectors, rtol=0, atol=0)
    assert (original / 'modules.json').read_bytes() == module_config
    # A native ST pipeline plus this tenant adapter preserves Dense/Pooling/Normalize.
    reference[0].model = PeftModel.from_pretrained(reference[0].model, checkpoint)
    reference.eval()
    with torch.no_grad():
        reloaded = reference(dict(features))['sentence_embedding']
    torch.testing.assert_close(reloaded, saved_vectors, rtol=0, atol=1e-6)
    model.remove_adapter('b')
    model.remove_adapter('c')
