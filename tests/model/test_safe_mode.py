"""Exercise a real CPU forward through the Ray-mode decorators."""
from types import SimpleNamespace

import pytest
import torch

import twinkle.infra as infra
from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor
from twinkle import remote_function


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


@pytest.mark.parametrize('mode', ['local', 'ray'])
@pytest.mark.parametrize('decorated', [False, True])
def test_custom_processor_entry_point_is_preserved(monkeypatch, mode, decorated):
    monkeypatch.setattr(infra, '_mode', 'local')
    model, group = _cpu_model()

    class CustomProcessor(InputProcessor):
        def __call__(self, inputs, **kwargs):
            self.seen_model = kwargs['model']
            return super().__call__(inputs, **kwargs)

    if decorated:
        CustomProcessor.__call__ = remote_function()(CustomProcessor.__call__)
    processor = object.__new__(CustomProcessor)
    processor.__dict__.update(group.processor.__dict__)
    processor.seen_model = None
    model.device_mesh = None
    model.set_processor(processor)
    monkeypatch.setattr(infra, '_mode', mode)
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    model.forward(inputs={'input_ids': [1, 2], 'labels': [2, 3]})
    assert processor.seen_model is model.model
    if mode == 'ray':
        with pytest.raises(ValueError, match='Callable or Type'):
            InputProcessor.__call__(processor, {'input_ids': [1]}, external=lambda: None)


def test_local_processor_scope_does_not_cover_other_objects(monkeypatch):
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    _, group = _cpu_model()
    _, other = _cpu_model()

    def pipeline(inputs, **kwargs):
        other.processor(inputs, external=lambda: None)

    group.processor.process_pipeline = [pipeline]
    with pytest.raises(ValueError, match='Callable or Type'):
        group.processor._process({'input_ids': [1]}, model=lambda: None)
    with pytest.raises(ValueError, match='Callable or Type'):
        group.processor({'input_ids': [1]}, external=lambda: None)


def test_local_processor_call_rejects_remote_handles():
    _, group = _cpu_model()
    group.processor._actors = []
    with pytest.raises(ValueError, match='remote handle'):
        group.processor._process({'input_ids': [1]})
