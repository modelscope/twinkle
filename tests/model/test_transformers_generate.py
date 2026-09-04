from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _Template:
    tokenizer = _Tokenizer()

    def batch_encode(self, rows, add_generation_prompt=False):
        assert add_generation_prompt
        return [
            {
                'input_ids': np.array([10, 11]),
                'attention_mask': np.ones(2),
                'position_ids': np.arange(2),
                'labels': np.array([-100, -100]),
            } for _ in rows
        ]

    def decode(self, token_ids, **kwargs):
        return ' '.join(map(str, token_ids))


class _GenerateModel:

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        completion = torch.tensor([[20, 2]], device=input_ids.device).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, completion], dim=-1)


class _Strategy:

    def __init__(self):
        self.generation_events = []

    @staticmethod
    def unwrap_model(model):
        return model

    @contextmanager
    def generation_context(self, model):
        self.generation_events.append(('enter', model))
        try:
            yield
        finally:
            self.generation_events.append(('exit', model))


def _model_wrapper():
    wrapper = object.__new__(TransformersModel)
    wrapper.optimizer_group = {
        'tenant': SimpleNamespace(template=_Template(), processor=InputProcessor()),
    }
    wrapper.model = _GenerateModel()
    wrapper.strategy = _Strategy()
    wrapper.hf_config = SimpleNamespace()
    wrapper._model_wrapped = True
    wrapper._enable_sp = False
    wrapper._lazy_wrap_model = lambda: None
    wrapper._get_default_group = lambda: 'tenant'
    return wrapper


def test_generate_encodes_trajectory_and_returns_completion_only():
    wrapper = _model_wrapper()
    torch.manual_seed(1234)
    rng_state = torch.get_rng_state().clone()
    with patch('twinkle.processor.base.Platform.get_local_device', return_value=torch.device('cpu')):
        result = TransformersModel.generate.__wrapped__(
            wrapper,
            inputs=[{'messages': [{'role': 'user', 'content': 'hello'}]}],
            adapter_name='tenant',
            generation_config={'max_new_tokens': 2},
        )

    assert result == [{
        'prompt_token_ids': [10, 11],
        'tokens': [20, 2],
        'text': '20 2',
        'stop_reason': 'stop',
    }]
    assert wrapper.strategy.generation_events == [
        ('enter', wrapper.model),
        ('exit', wrapper.model),
    ]
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_generated_token_ids_trims_padding_after_eos():
    tokens, stopped = TransformersModel._generated_token_ids(
        torch.tensor([10, 11, 20, 2, 0, 0]),
        prompt_width=2,
        eos_token_ids=[2],
        pad_token_id=0,
    )
    assert tokens == [20, 2]
    assert stopped is True
