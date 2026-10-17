# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU training through production methods, processors and Ray-mode decorators."""
from types import SimpleNamespace

import pytest
import torch

import twinkle.infra as infra
from twinkle.loss import CrossEntropyLoss
from twinkle.model.transformers.transformers import OptimizerGroup, TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import Platform


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)

    def forward(self, input_ids, **kwargs):
        return {'logits': self.embedding(input_ids)}


@pytest.fixture
def cpu_model(monkeypatch):
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '0')
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setattr(Platform, 'get_local_device', lambda *args, **kwargs: torch.device('cpu'))

    def build(model_cls=TransformersModel):
        # Construct worker-owned objects without Ray driver handles or tokenizer
        # downloads. Keep the full processor pipeline, loss and backward methods.
        with monkeypatch.context() as local:
            local.setattr(infra, '_mode', 'local')
            processor = InputProcessor(padding_free=False)
        group = OptimizerGroup(
            processor=processor,
            template=object.__new__(Template),
            loss_instance=CrossEntropyLoss(),
        )
        model = object.__new__(model_cls)
        torch.nn.Module.__init__(model)
        model.model = _TinyModel()
        model.optimizer_group = {'': group}
        model.device_mesh = None
        model.mixed_precision = 'no'
        model.sp_strategy = None
        model.hf_config = SimpleNamespace()
        model._model_wrapped = True
        model._router_replay_setup = lambda *args: lambda: None
        model.strategy = SimpleNamespace(unwrap_model=lambda obj: obj)
        return model, group

    return build


def _features():
    return [{'input_ids': [1, 2, 3, 4], 'labels': [2, 3, 4, 5]}]


def _assert_gradients(model):
    grad = model.model.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.norm().item() > 0


@pytest.mark.parametrize('mode', ['local', 'ray'])
@pytest.mark.parametrize('trust', ['0', '1'])
@pytest.mark.parametrize('entry', ['forward_backward', 'micro_batch', 'forward_only'])
def test_training_entries(cpu_model, monkeypatch, mode, trust, entry):
    monkeypatch.setattr(infra, '_mode', mode)
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', trust)
    model, _ = cpu_model()
    if entry == 'forward_only':
        model.forward_only(inputs=_features())
        loss = model.calculate_loss()
        assert model.model.embedding.weight.grad is None
    else:
        kwargs = {'micro_batch_size': 1} if entry == 'micro_batch' else {}
        inputs = _features() * 2 if kwargs else _features()
        loss = model.forward_backward(inputs=inputs, **kwargs)['loss']
        _assert_gradients(model)
    assert torch.isfinite(torch.tensor(loss))


@pytest.mark.parametrize('trust', ['0', '1'])
def test_tinker_forward_backward(cpu_model, monkeypatch, trust):
    import numpy as np
    types = pytest.importorskip('tinker').types
    from twinkle.server.model.backends.transformers_model import TwinkleCompatFullTransformersModel

    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', trust)
    model, _ = cpu_model(model_cls=TwinkleCompatFullTransformersModel)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={
            'target_tokens': types.TensorData.from_numpy(np.array([2, 3, 4, 5], dtype=np.int64)),
            'weights': types.TensorData.from_numpy(np.ones(4, dtype=np.float32)),
        },
    )
    results, loss = model.tinker_forward_backward(inputs=[datum], adapter_name='', loss_fn='cross_entropy')
    assert results and torch.isfinite(torch.tensor(loss))
    _assert_gradients(model)


@pytest.mark.parametrize('value', [lambda: None, type, {'nested': [lambda: None]}])
@pytest.mark.parametrize('remote_handle', [False, True])
def test_public_processor_rejects_callable(cpu_model, value, remote_handle):
    _, group = cpu_model()
    if remote_handle:
        group.processor._actors = []
    with pytest.raises(ValueError, match='Callable or Type'):
        group.processor(_features(), external=value)


def test_internal_processing_keeps_public_safety_checks(cpu_model):
    _, group = cpu_model()
    _, other = cpu_model()

    def pipeline(inputs, **kwargs):
        other.processor(inputs, external=lambda: None)

    group.processor.process_pipeline = [pipeline]
    with pytest.raises(ValueError, match='Callable or Type'):
        group.processor.process_local(_features(), model=lambda: None)


def test_local_call_rejects_remote_handle(cpu_model):
    _, group = cpu_model()
    group.processor._actors = []
    with pytest.raises(ValueError, match='remote handle'):
        group.processor.process_local(_features())


def test_internal_processor_entry_is_not_exported():
    assert not hasattr(InputProcessor.process_local, '_execute')
