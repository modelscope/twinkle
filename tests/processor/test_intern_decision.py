# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU decision encoding and full-vocabulary loss/gradient regressions."""
import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import CrossEntropyLoss


@pytest.fixture
def adapter():
    root = Path(__file__).resolve().parents[2] / 'cookbook/transformers/intern_decision_npu/twinkle_adapter'
    modules = {}
    for name in ('data', 'processor'):
        spec = importlib.util.spec_from_file_location(f'decision_test_{name}', root / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 9, 3, 9, 4]

    def encode(self, text, **kwargs):
        return {'<decision>': [9], 'A': [5], 'B': [6], 'invalid': [7, 8]}[text]


def test_single_shift_and_no_answer_leakage(adapter):
    record = {'messages': [{'role': 'assistant', 'content': 'skeleton'}], 'decision_targets': ['A', 'B']}
    feature = adapter.data.encode_decision(Tokenizer(), record)
    assert feature['labels'] == [-100, 5, -100, 6, -100, -100]
    changed = deepcopy(record)
    changed['decision_targets'] = ['B', 'A']
    assert adapter.data.encode_decision(Tokenizer(), changed)['input_ids'] == feature['input_ids']
    assert adapter.data.count_supervised_tokens([feature]) == 2
    tensor_row = {'labels': torch.tensor(feature['labels']).unsqueeze(0)}
    assert adapter.data.count_supervised_tokens([tensor_row]) == 2


@pytest.mark.parametrize('targets', [[], ['A'], ['A', 'invalid']])
def test_invalid_targets(adapter, targets):
    with pytest.raises(ValueError):
        adapter.data.encode_decision(Tokenizer(), {'messages': [], 'decision_targets': targets})


def test_no_truncation(adapter):
    with pytest.raises(ValueError, match='truncate'):
        adapter.data.encode_decision(Tokenizer(), {'messages': [], 'decision_targets': ['A', 'B']}, max_length=5)


def test_selected_loss_and_gradient(adapter):
    processor = adapter.processor.DecisionProcessor(pad_token_id=0)
    b, t, v = 3, 12, 17
    unshifted = torch.full((b, t), -100, dtype=torch.long)
    unshifted[0, 4], unshifted[1, 7], unshifted[2, 3], unshifted[2, 8] = 2, 5, 9, 11
    shifted = torch.full_like(unshifted, -100)
    shifted[:, :-1] = unshifted[:, 1:]
    logits = torch.randn(b, t, v, generator=torch.Generator().manual_seed(42),
                         dtype=torch.float64, requires_grad=True)
    reference = F.cross_entropy(logits[:, :-1].reshape(-1, v), unshifted[:, 1:].reshape(-1))
    selected = processor.select_decision_logits({'input_ids': torch.ones(b, t, dtype=torch.long),
                                                 'attention_mask': torch.ones(b, t, dtype=torch.long),
                                                 'labels': shifted})
    assert selected['input_ids'].shape[-1] == 128
    assert not selected['use_cache']
    sparse = logits.index_select(1, selected['logits_to_keep'])
    labels = selected['labels']
    logps = sparse.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    actual = CrossEntropyLoss(reduction='mean')(selected, {'logps': logps})['loss']
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    r = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(a, r, rtol=1e-12, atol=1e-12)


def test_missing_supervision_rejected(adapter):
    processor = adapter.processor.DecisionProcessor(pad_token_id=0)
    with pytest.raises(ValueError, match='supervision'):
        processor.select_decision_logits({'labels': torch.full((1, 5), -100)})
