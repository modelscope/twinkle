"""Exercise a real CPU forward through the Ray-mode decorators."""
from types import SimpleNamespace

import pytest
import torch

import twinkle.infra as infra
from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)

    def forward(self, input_ids):
        return {'logits': self.embedding(input_ids)}


def _cpu_model():
    processor = object.__new__(InputProcessor)
    processor.process_pipeline = [lambda inputs, **kwargs: {k: torch.tensor([v]) for k, v in inputs.items()}]
    processor.postprocess_tensor_sp = lambda inputs, outputs, **kwargs: (inputs, outputs)
    processor.unpack_packed_sequences = lambda inputs, outputs, **kwargs: (inputs, outputs)
    status = SimpleNamespace()
    group = SimpleNamespace(processor=processor, loss_instance=CrossEntropyLoss(), train_status=status,
                            accumulate_metrics=lambda _: None)
    model = object.__new__(TransformersModel)
    torch.nn.Module.__init__(model)
    model.model = _TinyModel()
    model.optimizer_group = {'': group}
    model.sp_strategy = None
    model.hf_config = SimpleNamespace()
    model._get_default_group = lambda: ''
    model._lazy_wrap_model = lambda: None
    model._router_replay_setup = lambda *args: lambda: None
    return model, group


def test_safe_mode_real_forward_and_backward(monkeypatch):
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    model, group = _cpu_model()
    outputs = model.forward(inputs={'input_ids': [1, 2], 'labels': [2, 3]})
    loss = group.loss_instance(group.train_status.inputs, outputs)['loss']
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.model.embedding.weight.grad).all()


@pytest.mark.parametrize('value', [lambda: None, type, {'nested': [lambda: None]}])
def test_safe_mode_public_processor_rejects_callable(monkeypatch, value):
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    model, group = _cpu_model()
    with pytest.raises(ValueError, match='Callable or Type'):
        group.processor({'input_ids': [1]}, external=value)
