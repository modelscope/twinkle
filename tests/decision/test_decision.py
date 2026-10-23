# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import json

import pytest
import torch

from twinkle.loss.decision import DecisionLoss
from twinkle.model.decision.encoding import DecisionEncoder
from twinkle.model.decision.heads import ClefJointSchemaHead, JevVerbalizerHead
from twinkle.model.decision.model import activate_decision_head, attach_decision_head, decision_lora_config
from twinkle.preprocessor.decision import ClefPreprocessor, JevPreprocessor, OmniJevPreprocessor
from twinkle.processor.decision import DecisionInputProcessor


@pytest.mark.parametrize('mode', ['scoring', 'distill_rps', 'smooth_brier', 'ce_rps'])
def test_loss_and_gradients_against_unpadded_reference(mode):
    raw = torch.tensor([[0.8, -0.4, 91.0], [0.2, -0.5, 0.7]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    target = torch.tensor([[.3, .7, 0], [.2, .3, .5]], dtype=torch.float64)
    actual = DecisionLoss(mode)({'labels': target}, {'logits': raw, 'option_mask': mask,
                                                  'kinds': torch.tensor([0, 2])})
    expected = 0
    for i, n in enumerate([2, 3]):
        p = raw[i, :n].softmax(-1)
        t = target[i, :n]
        smooth = .9 * t + .1 / n if mode == 'smooth_brier' else t
        expected = expected - (smooth * p.log()).sum()
        if mode == 'smooth_brier':
            expected = expected + (p - t).square().mean()
        if i == 1 and mode in ('distill_rps', 'ce_rps'):
            rps = (p.cumsum(0)[:-1] - t.cumsum(0)[:-1]).square().sum()
            expected = expected + .5 * rps / (n - 1 if mode == 'distill_rps' else 1)
    torch.testing.assert_close(actual['loss'], expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual['loss'], raw, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(expected, raw)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-12)
    assert actual_grad[0, 2] == 0
    assert actual['num_tokens'] == 2


@pytest.mark.parametrize('target', [[.3, .4], [-.1, 1.1], [float('nan'), 0.]])
def test_invalid_targets_fail(target):
    with pytest.raises(ValueError):
        DecisionLoss()({'labels': torch.tensor([target])}, {
            'logits': torch.zeros(1, 2), 'option_mask': torch.ones(1, 2, dtype=torch.bool)})


def test_uneven_microbatches_have_identical_normalized_gradients():
    z = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    target = torch.eye(3, dtype=torch.float64)[torch.tensor([0, 2, 1, 0, 1])]
    loss = DecisionLoss()
    def part(start, end):
        return loss({'labels': target[start:end]}, {'logits': z[start:end],
                    'option_mask': torch.ones(end-start, 3, dtype=torch.bool)})
    full = part(0, 5)
    a, b = part(0, 1), part(1, 5)
    g1 = torch.autograd.grad(full['loss']/full['num_tokens'], z, retain_graph=True)[0]
    g2 = torch.autograd.grad((a['loss']+b['loss'])/(a['num_tokens']+b['num_tokens']), z)[0]
    torch.testing.assert_close(g1, g2, rtol=1e-12, atol=1e-12)


class Tokens:
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        symbols = ['false', 'true'] + list('012345') + list('ABCDEFGHIJKLMNOP')
        if text in symbols:
            return [symbols.index(text) + 1]
        return [ord(c) % 31 + 25 for c in text]


def records():
    return [{'state': 'sky', 'questions': [{'kind': 'noul', 'question': 'Blue?', 'gold': 1}]},
            {'state': 'water', 'questions': [{'kind': 'choice', 'question': 'Pick?',
                                           'options': ['cold', 'warm', 'hot'], 'gold': 0}]}]


def test_encode_excludes_gold_and_collates_without_mutation():
    pre = JevPreprocessor()
    encoder = DecisionEncoder(Tokens())
    normalized, dropped = pre(records())
    encoded = [encoder(r) for r in normalized]
    changed = copy.deepcopy(normalized[0]); changed['target_probs'] = [[1., 0.]]
    assert encoder(changed)['input_ids'] == encoded[0]['input_ids']
    before = copy.deepcopy(encoded)
    for side in ['left', 'right']:
        batch = DecisionInputProcessor(0, padding_side=side)(encoded)
        assert batch['labels'].shape == (2, 3)
        assert batch['decision_meta']['counts'] == [1, 1]
        assert batch['decision_meta']['left_padding'] == (side == 'left')
    assert encoded == before
    with pytest.raises(ValueError):
        DecisionEncoder(Tokens(), max_length=3)(normalized[0])


def test_clef_joint_spans_and_labels():
    row = {'state': 'sky', 'questions': {
        'blue': {'type': 'noul', 'instructions': 'Blue?'},
        'quality': {'type': 'score', 'instructions': 'Quality?', 'criteria': ['bad', 'good']}},
        'answers': {'blue': True, 'quality': 1}}
    normalized, _ = ClefPreprocessor()([row])
    encoded = DecisionEncoder(Tokens(), 'clef')(normalized[0])
    assert len(encoded['decision_meta']['kinds']) == 2
    for start, end in encoded['decision_meta']['question_spans']:
        assert 0 <= start < end <= len(encoded['input_ids'])
    assert len(encoded['target_probs']) == 2


def test_omnijev_schema_and_raw_video_rejection():
    row = {'state': 'sky', 'questions': [{'type': 'choice', 'instructions': 'Pick',
          'options': [{'key': 'blue', 'text': 'blue'}, {'key': 'none', 'abstain': True}], 'gold': 'blue'}]}
    normalized, _ = OmniJevPreprocessor()([row])
    assert sum(normalized[0]['target_probs'][0]) == 1
    with pytest.raises(ValueError):
        OmniJevPreprocessor()([dict(row, videos=['raw.mp4'])])


def test_clef_embedding_is_not_owned_or_copied():
    embedding = torch.nn.Linear(8, 64, bias=False)
    head = ClefJointSchemaHead(8, 8, 1, 1, 2, 16)
    head.set_output_embeddings(embedding)
    other = copy.deepcopy(head)
    assert other._output_embeddings is embedding
    assert all('_output_embeddings' not in key for key in head.state_dict())


def test_lora_and_head_update_save_reload(tmp_path):
    from peft import PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    base = LlamaForCausalLM(config)
    initial = copy.deepcopy(base.state_dict())
    attach_decision_head(base, Tokens())
    model = get_peft_model(base, decision_lora_config(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj']))
    activate_decision_head(model)
    normalized, _ = JevPreprocessor()(records())
    encoded = [DecisionEncoder(Tokens())(r) for r in normalized]
    batch = DecisionInputProcessor(0)(encoded, model=model)
    labels = batch.pop('labels')
    trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any('scoring_head' in n for n in trainable)
    assert all('scoring_head' in n or 'lora_' in n for n in trainable)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        out = model(**batch)
        value = DecisionLoss()({'labels': labels}, out)
        (value['loss']/value['num_tokens']).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step(); optimizer.zero_grad()
    assert any(not torch.equal(p, trainable[n]) for n,p in model.named_parameters()
               if n in trainable and 'scoring_head' in n)
    assert any(not torch.equal(p, trainable[n]) for n,p in model.named_parameters()
               if n in trainable and 'lora_' in n)
    model.eval()
    with torch.no_grad():
        expected = model(**batch).logits
    model.save_pretrained(tmp_path)
    restored = LlamaForCausalLM(config); restored.load_state_dict(initial)
    attach_decision_head(restored, Tokens())
    restored = PeftModel.from_pretrained(restored, tmp_path)
    restored.eval()
    with torch.no_grad():
        actual = restored(**batch).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multitoken_verbalizer_rejected():
    class BadTokens(Tokens):
        def encode(self, *args, **kwargs):
            return [1, 2]
    with pytest.raises(ValueError):
        JevVerbalizerHead.from_tokenizer(BadTokens())


def test_accuracy_counts_questions_and_records():
    from twinkle.metric.decision import DecisionAccuracy
    metric = DecisionAccuracy()
    inputs = {'labels': torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
              'decision_meta': {'counts': [2, 1]}}
    metric.accumulate(inputs, {'logits': torch.tensor([[2., 1.], [2., 1.], [2., 1.]]),
                              'option_mask': torch.ones(3, 2, dtype=torch.bool)})
    result = metric.calculate()
    assert result['acc'] == 2/3 and result['record_acc'] == .5
    assert metric.calculate() == {}


@pytest.mark.parametrize('family,expected', [('jev', [.2, .8]), ('clef', [.8, .2]), ('omnijev', [.8, .2])])
def test_open_jev_boolean_option_order(family, expected):
    from twinkle.preprocessor.decision import OpenJevPreprocessor
    cls = {'jev': JevPreprocessor, 'clef': ClefPreprocessor, 'omnijev': OmniJevPreprocessor}[family]
    rows, _ = OpenJevPreprocessor(cls())([{'kind': 'noul', 'question': 'Blue?', 'state': 'sky',
                                         'options': ['yes', 'no'], 'target': [.8, .2]}])
    assert rows[0]['target_probs'][0] == expected


def test_omnijev_head_three_types_and_gradients():
    from twinkle.model.decision.heads import OmniJevHead
    head = OmniJevHead(hidden_size=8, head_hidden=16, ord_hidden=8,
                       norm='softmax', lm_feats=True, ordinal=True)
    features = [(torch.randn(n, 8, requires_grad=True), torch.randn(8, requires_grad=True), torch.randn(n, 6))
                for n in [1, 2, 3]]
    meta = {'kinds': [0, 1, 2], 'n_options': [2, 3, 3], 'counts': [1, 1, 1], 'max_opt': 3}
    output = head.forward_features(features, meta)
    labels = torch.tensor([[1., 0., 0.], [.1, .2, .7], [.2, .4, .4]])
    result = DecisionLoss('ce_rps')({'labels': labels}, output)
    result['loss'].backward()
    assert all(u.grad is not None and torch.isfinite(u.grad).all() for u, q, f in features)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.ord.parameters())
    assert output.option_mask.sum(-1).tolist() == [2, 3, 3]


def test_clef_joint_head_forward_padding_invariance_and_gradients():
    torch.manual_seed(8)
    head = ClefJointSchemaHead(8, 8, 1, 1, 2, 16)
    embedding = torch.nn.Linear(8, 64, bias=False)
    embedding.requires_grad_(False)
    head.set_output_embeddings(embedding)
    hidden = torch.randn(1, 10, 8, requires_grad=True)
    ids = torch.arange(10)[None]
    meta = {'input_ids': ids, 'record_index': [0, 0], 'seq_lens': [10], 'kinds': [0, 1],
            'n_options': [2, 3], 'counts': [2], 'max_opt': 3, 'question_spans': [(0, 1), (1, 2)],
            'option_spans': [[(2, 3), (3, 4)], [(4, 5), (5, 6), (6, 7)]]}
    output = head(hidden, None, meta)
    padded = torch.cat([torch.zeros(1, 2, 8), hidden], 1)
    left_meta = dict(meta, input_ids=torch.cat([torch.zeros(1, 2, dtype=torch.long), ids], 1), left_padding=True)
    reference = head(padded, None, left_meta)
    torch.testing.assert_close(output.logits, reference.logits, rtol=0, atol=0)
    loss = DecisionLoss('smooth_brier')({'labels': torch.tensor([[1., 0., 0.], [0., 1., 0.]])}, output)
    loss['loss'].backward()
    assert torch.isfinite(hidden.grad).all()
    assert embedding.weight.grad is None


def test_jev_image_processor_payload(tmp_path):
    from PIL import Image
    path = tmp_path / 'red.png'; Image.new('RGB', (32, 32), 'red').save(path)
    class VisionProcessor:
        tokenizer = Tokens()
        image_processor = object()
        def __call__(self, text, images, return_tensors):
            assert '<|image_pad|>' in text[0] and len(images) == 1
            return {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3),
                    'pixel_values': torch.zeros(4, 12), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    normalized, _ = JevPreprocessor()([dict(records()[0], images=[str(path)])])
    row = DecisionEncoder(VisionProcessor(), 'jev')(normalized[0])
    batch = DecisionInputProcessor(0)([row, row])
    assert batch['pixel_values'].shape == (8, 12)
    assert batch['image_grid_thw'].shape == (2, 3)
    assert batch['decision_meta']['mm_counts'] == [{'image': 1, 'video': 0}] * 2


def test_jev_calibration_next_to_head(tmp_path):
    import json
    from twinkle.model.decision.heads import JevVerbalizerHead
    (tmp_path / 'decision_head.json').write_text(json.dumps({
        'bias': [0.] * 24, 'verbalizer_ids': list(range(24))}))
    (tmp_path / 'calibration.json').write_text(json.dumps({'per_kind': {'choice': 2.5}}))
    head = JevVerbalizerHead.from_pretrained(str(tmp_path))
    assert head.temperatures[1].item() == 2.5


def test_reject_duplicate_verbalizers():
    from twinkle.model.decision.heads import JevVerbalizerHead
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1]
    with pytest.raises(ValueError, match='distinct'):
        JevVerbalizerHead.from_tokenizer(Tokenizer())


def test_reject_unknown_verbalizer():
    from twinkle.model.decision.heads import JevVerbalizerHead
    class Tokenizer:
        unk_token_id = 1
        def encode(self, text, **kwargs):
            return [1]
    with pytest.raises(ValueError, match='one token'):
        JevVerbalizerHead.from_tokenizer(Tokenizer())


def test_omnijev_text_state_reaches_prompt():
    class Processor:
        tokenizer = Tokens()
        seen = None
        def apply_chat_template(self, messages, **kwargs):
            self.seen = messages
            return messages[0]['content'][0]['text'] + messages[1]['content']
    processor = Processor()
    row = {'state': 'The secret context is BLUE.', 'questions': [{'type': 'noul',
            'instructions': 'Is the context blue?', 'gold': True}]}
    normalized, _ = OmniJevPreprocessor()([row])
    DecisionEncoder(processor, 'omnijev')(normalized[0])
    assert 'The secret context is BLUE.' in processor.seen[0]['content'][0]['text']


def test_released_adapter_without_head_uses_factory_head():
    from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import LlamaConfig, LlamaForCausalLM
    from twinkle.model.decision.model import initialize_missing_adapter_head
    config = LlamaConfig(vocab_size=64, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    model = LlamaForCausalLM(config)
    attach_decision_head(model, Tokens())
    model = get_peft_model(model, decision_lora_config(r=2, target_modules=['q_proj']), adapter_name='decision')
    state = get_peft_model_state_dict(model, adapter_name='decision')
    released = {k: v for k, v in state.items() if '.scoring_head.' not in k}
    prepared = initialize_missing_adapter_head(model, released, 'decision')
    loaded = set_peft_model_state_dict(model, prepared, adapter_name='decision')
    assert not loaded.unexpected_keys
    assert not any('scoring_head.modules_to_save.decision' in k for k in loaded.missing_keys)
    with pytest.raises(ValueError):
        initialize_missing_adapter_head(model, state, 'decision')


def test_clef_option_subsampling_keeps_schema_and_target_aligned():
    from twinkle.preprocessor.decision import DecisionAugmentConfig
    pre = ClefPreprocessor(augment_config=DecisionAugmentConfig(subsample_options=2))
    row = {'state': 'test', 'questions': {'q': {'type': 'choice', 'instructions': 'pick',
            'criteria': {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}}}, 'answers': {'q': 'd'}}
    normalized, _ = pre([row])
    result = normalized[0]
    criteria = json.loads(result['criteria'][0])
    assert len(criteria) == len(result['options'][0]) == 2
    assert result['options'][0][result['target_probs'][0].index(1.)] == 'd'
    encoded = DecisionEncoder(Tokens(), 'clef')(result)
    assert encoded['decision_meta']['n_options'] == [2]


@pytest.mark.parametrize('gold', ['d', 'abstain'])
def test_omnijev_option_subsampling_retains_abstain_and_target(gold):
    from twinkle.preprocessor.decision import DecisionAugmentConfig
    pre = OmniJevPreprocessor(augment_config=DecisionAugmentConfig(subsample_options=2))
    row = {'questions': [{'type': 'choice', 'instructions': 'pick',
            'options': [{'key': k, 'text': k} for k in 'abcd'], 'gold': gold}]}
    normalized, _ = pre([row])
    result = normalized[0]
    assert len(result['options'][0]) == 2
    keys = result['omnijev_column_keys'][0]
    assert len(keys) == len(result['target_probs'][0]) == 3
    assert keys[-1] == 'abstain'
    assert keys[result['target_probs'][0].index(1.)] == gold
