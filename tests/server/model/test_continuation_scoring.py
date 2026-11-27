# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from twinkle.protocol.types.model import ForwardOnlyRequest
from twinkle.server.config.training import TrainingArguments
from twinkle.server.model.backends.common import TwinkleCompatModelBase
from twinkle_client._request_builder import build_request


def test_training_template_requires_class_before_model_allocation():
    with pytest.raises(ValidationError, match='template requires a nonempty template_cls'):
        TrainingArguments(template={'enable_thinking': False})
    template = {'template_cls': 'Qwen3_5Template', 'enable_thinking': False}
    assert TrainingArguments(template=template).template == template


class TextTemplate:
    """012aff34 complete-Trajectory contract: encode_prompt/append_completion exchange plain
    dict features (input_ids + output-order labels), not .state wrappers."""

    tokenizer = range(100)
    truncation_strategy = 'left'

    def encode_prompt(self, row, **kwargs):
        assert self.truncation_strategy == 'delete'
        ids = [12] * len(row['messages'][-1]['content'])
        return {'input_ids': ids, 'labels': [-100] * len(ids)}

    def append_completion(self, prompt, tokens, **kwargs):
        prefix = list(prompt['input_ids'])
        return {'input_ids': prefix + list(tokens),
                'labels': [-100] * (len(prefix) - 1) + list(tokens) + [-100]}


def test_continuation_is_encoded_once_and_keeps_row_identity():
    prompts = [
        {'messages': [{'role': 'user', 'content': question}], 'episode_id': label}
        for question, label in [('Short', 'a'), ('Much longer question', 'b')]
    ]
    request = build_request(
        ForwardOnlyRequest,
        training_id='training-a',
        adapter_name='adapter',
        inputs=prompts,
        continuation_token_ids=[[33, 45], [33, 45]],
        output_frame='response',
        output_fields=['logps'],
    )
    assert request.loss_kwargs == {'output_frame': 'response', 'output_fields': ['logps']}

    template = TextTemplate()
    model = TwinkleCompatModelBase()
    model.optimizer_group = {'adapter': SimpleNamespace(template=template)}
    result = model.encode_scoring_inputs(prompts, request.continuation_token_ids, adapter_name='adapter')
    assert [row['episode_id'] for row in result] == ['a', 'b']
    for row, prompt in zip(result, prompts, strict=True):
        tokens = [33, 45]
        start = len(prompt['messages'][0]['content'])
        assert row['input_ids'][start:] == row['labels'][start - 1:-1] == row['response_token_ids'] == tokens
    assert template.truncation_strategy == 'left'


@pytest.mark.parametrize('tokens', [[], [100], [-1]])
def test_invalid_continuation_is_rejected_before_forward(tokens):
    model = TwinkleCompatModelBase()
    model.optimizer_group = {'adapter': SimpleNamespace(template=TextTemplate())}
    with pytest.raises(ValueError, match='configured vocabulary'):
        model.encode_scoring_inputs(
            [{'messages': [{'role': 'user', 'content': 'hello'}]}], [tokens], adapter_name='adapter')
