"""Text decision encoding; Twinkle consumes labels already shifted for causal CE."""
from collections.abc import Mapping
from copy import deepcopy

import torch


def count_supervised_tokens(features):
    """Count labels in raw rows and in the dataloader's singleton tensor rows."""
    return sum(int(torch.as_tensor(row['labels']).ne(-100).sum().item()) for row in features)


def encode_decision(tokenizer, record, max_length=8192):
    messages = deepcopy(record['messages'])
    if any(not isinstance(m['content'], str) for m in messages):
        raise ValueError('Only text decisions are supported by this training adapter')
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=False, enable_thinking=False)
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    ids = list(ids)
    if len(ids) > max_length:
        raise ValueError('Refusing to truncate decision evidence')
    marker = tokenizer.encode('<decision>', add_special_tokens=False)
    if len(marker) != 1:
        raise ValueError('Register <decision> as one special token first')
    positions = [i for i, token in enumerate(ids) if token == marker[0]]
    targets = record['decision_targets']
    if not positions or len(positions) != len(targets):
        raise ValueError('Decision marker/target mismatch')
    labels = [-100] * len(ids)
    for position, symbol in zip(positions, targets):
        target = tokenizer.encode(symbol, add_special_tokens=False)
        if position == 0 or len(target) != 1:
            raise ValueError('Decision requires one answer token and a preceding position')
        # Unlike HF causal loss, Twinkle's external CE does NOT shift labels.
        labels[position - 1] = target[0]
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}
