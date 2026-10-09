"""CPU checks against Twinkle's actual processor and CE, with real tokenizer."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from twinkle.loss import CrossEntropyLoss
from .data import count_supervised_tokens, encode_decision
from .processor import DecisionProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/workspace/results/twinkle-cpu-contract.json'))
    args = parser.parse_args()
    raw = [{'labels': [-100, 2, -100]}, {'labels': [3, -100, 5, -100]}]
    tensors = [{'labels': torch.tensor(row['labels']).unsqueeze(0)} for row in raw]
    assert count_supervised_tokens(raw) == count_supervised_tokens(tensors) == 3
    tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-4B')
    tokenizer.add_special_tokens({'additional_special_tokens': ['<decision>']})
    processor = DecisionProcessor(pad_token_id=tokenizer.pad_token_id)
    count = 0
    decisions = 0
    sample_features = []
    for split in ['train', 'validation', 'calibration', 'test']:
        for line in Path('/data/joint', split + '.jsonl').read_text().splitlines():
            record = json.loads(line)
            feature = encode_decision(tokenizer, record)
            assert sum(x != -100 for x in feature['labels']) == len(record['decision_targets'])
            changed = deepcopy(record)
            changed['decision_targets'] = ['Z'] * len(record['decision_targets'])
            assert encode_decision(tokenizer, changed)['input_ids'] == feature['input_ids']
            if len(sample_features) < 4:
                sample_features.append(feature)
            count += 1
            decisions += len(record['decision_targets'])
    # Call the real processor on CPU in a container with no accelerator devices.
    # Twinkle's platform defaults to CUDA when neither accelerator exists.
    # Override device placement ONLY in this CPU numerical test.
    with patch('twinkle.processor.base.Platform.get_local_device', return_value='cpu'):
        batch = processor(deepcopy(sample_features))
    assert batch['input_ids'].shape[-1] % 128 == 0
    assert batch['use_cache'] is False
    assert batch['labels'].shape[-1] == len(batch['logits_to_keep'])
    assert int((batch['labels'] != -100).sum()) == sum(sum(x != -100 for x in f['labels']) for f in sample_features)
    # Smaller synthetic vocabulary gives an exact numerical/gradient comparison
    # without allocating a full sequence x Qwen vocabulary tensor on CPU.
    torch.manual_seed(42)
    b, t, v = 3, 12, 17
    unshifted = torch.full((b, t), -100, dtype=torch.long)
    unshifted[0, 4], unshifted[1, 7], unshifted[2, 3] = 2, 5, 9
    unshifted[2, 8] = 11  # Also cover multi-marker supervision.
    shifted = torch.full_like(unshifted, -100)
    shifted[:, :-1] = unshifted[:, 1:]
    logits = torch.randn(b, t, v, dtype=torch.float64, requires_grad=True)
    reference = F.cross_entropy(logits[:, :-1].reshape(-1, v),
                                unshifted[:, 1:].reshape(-1), ignore_index=-100)
    selected = processor.select_decision_logits({
        'input_ids': torch.ones(b, t, dtype=torch.long),
        'attention_mask': torch.ones(b, t, dtype=torch.long), 'labels': shifted})
    sparse = logits.index_select(1, selected['logits_to_keep'])
    labels = selected['labels']
    logps = sparse.log_softmax(-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    actual = CrossEntropyLoss(reduction='mean')(selected, {'logps': logps})['loss']
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    r = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(a, r, rtol=1e-12, atol=1e-12)
    result = {'status': 'passed', 'encoded_records': count, 'encoded_decisions': decisions,
              'label_shift': 'exactly_once_before_Twinkle_CE', 'gold_not_in_input': True,
              'real_processor': True, 'test_device_override': 'cpu', 'loss_and_gradients_match_causal_CE': True,
              'npu_training_tested': False}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
