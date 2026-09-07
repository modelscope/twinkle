# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from twinkle.loss import ChannelLoss, CrossEntropyLoss, torch_loss_mapping
from twinkle.metric import LossMetric
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


def _logps():
    return torch.tensor([
        [-0.2, -0.4, -0.6, -0.8],
        [-1.0, -1.2, -1.4, -1.6],
        [-0.1, -0.3, -0.5, -0.7],
    ])


def test_channel_loss_registered():
    assert torch_loss_mapping['channel'] is ChannelLoss


def test_cross_entropy_applies_loss_scale():
    labels = torch.tensor([[1, 2, -100], [1, 2, 3]])
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    loss_scale = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 2.0]])

    result = CrossEntropyLoss(reduction='sum')({
        'labels': labels,
        'loss_scale': loss_scale,
    }, {'logps': logps})

    assert result['loss'].item() == pytest.approx(19.0)
    assert result['num_tokens'].item() == 5


def test_channel_loss_keeps_cross_entropy_objective():
    labels = torch.tensor([[1, 2, -100, -100], [1, 2, 3, -100], [1, -100, -100, -100]])
    inputs = {'labels': labels, 'channel': ['math', 'code', 'math']}
    outputs = {'logps': _logps()}

    channel = ChannelLoss(reduction='sum')(inputs, outputs)
    reference = CrossEntropyLoss(reduction='sum')(inputs, outputs)

    assert torch.equal(channel['loss'], reference['loss'])
    assert torch.equal(channel['num_tokens'], reference['num_tokens'])
    assert torch.allclose(channel['channel_loss']['math'], torch.tensor([0.7, 3.0]))
    assert torch.allclose(channel['channel_loss']['code'], torch.tensor([3.6, 3.0]))


def test_channel_loss_defaults_to_none_and_applies_loss_scale():
    labels = torch.tensor([[1, 2, -100], [1, 2, 3]])
    logps = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    loss_scale = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 2.0]])

    result = ChannelLoss(reduction='sum')({
        'labels': labels,
        'loss_scale': loss_scale,
    }, {'logps': logps})

    assert result['loss'].item() == pytest.approx(19.0)
    assert torch.allclose(result['channel_loss'][None], torch.tensor([19.0, 5.0]))


def test_channel_loss_applies_dft_before_channel_aggregation():
    labels = torch.tensor([[1, 2], [3, -100]])
    logps = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    loss_scale = torch.tensor([[1.0, 0.5], [2.0, 0.0]])
    expected = -logps * logps.exp() * loss_scale

    result = ChannelLoss(reduction='sum', dft=True)({
        'labels': labels,
        'loss_scale': loss_scale,
        'channel': ['math', 'code'],
    }, {'logps': logps})

    assert torch.allclose(result['loss'], expected[0].sum() + expected[1, 0])
    assert torch.allclose(result['channel_loss']['math'], torch.stack((expected[0].sum(), torch.tensor(2.0))))
    assert torch.allclose(result['channel_loss']['code'], torch.stack((expected[1, 0], torch.tensor(1.0))))


def test_channel_loss_rejects_channel_count_mismatch():
    labels = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match='expected one per batch row'):
        ChannelLoss()({'labels': labels, 'channel': ['only-one']}, {'logps': _logps()[:2, :2]})


def test_loss_metric_aggregates_channels_and_resets():
    metric = LossMetric(device_mesh=None, process_group=None)
    metric.accumulate({}, {
        'loss': torch.tensor(4.0),
        'num_tokens': torch.tensor(2),
        'channel_loss': {
            'math': torch.tensor([3.0, 2.0]),
            'code': torch.tensor([4.0, 1.0]),
        },
    }, loss_reduction='sum')
    metric.accumulate({}, {
        'loss': torch.tensor(2.0),
        'num_tokens': torch.tensor(1),
        'channel_loss': {
            'math': torch.tensor([2.0, 1.0]),
        },
    }, loss_reduction='sum')

    result = metric.calculate()

    assert result['loss'] == '2.0000'
    assert result['loss_math'] == '1.6667'
    assert result['loss_code'] == '4.0000'
    assert metric.channel_loss == {}


def test_transformers_forward_keeps_loss_metadata_out_of_model_call():

    class PassthroughProcessor(InputProcessor):

        def __init__(self):
            pass

        def __call__(self, inputs, **kwargs):
            return dict(inputs)

        def postprocess_tensor_sp(self, inputs, outputs, **kwargs):
            return inputs, outputs

        def unpack_packed_sequences(self, inputs, outputs=None, **kwargs):
            return inputs, outputs

    class RecordingModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.received_keys = set()

        def forward(self, **kwargs):
            self.received_keys = set(kwargs)
            input_ids = kwargs['input_ids']
            return {'logits': torch.zeros((*input_ids.shape, 8))}

    model = object.__new__(TransformersModel)
    nn.Module.__init__(model)
    model.model = RecordingModel()
    model._model_wrapped = True
    model._router_replay_enabled = False
    model._enable_sp = False
    model.sp_strategy = None
    model.hf_config = None
    status = SimpleNamespace(inputs=None, outputs=None, forward_kwargs=None, loss_value=0)
    group = SimpleNamespace(
        processor=PassthroughProcessor(),
        loss_instance=ChannelLoss(reduction='sum'),
        template=None,
        train_status=status,
        accumulate_metrics=lambda _: None,
    )
    model.optimizer_group = {'': group}

    TransformersModel.forward.__wrapped__(model, inputs={
        'input_ids': torch.tensor([[1, 2]]),
        'labels': torch.tensor([[1, 2]]),
        'loss_scale': torch.ones(1, 2),
        'channel': ['math'],
    })

    assert model.model.received_keys == {'input_ids'}
    assert status.inputs['channel'] == ['math']
    assert torch.equal(status.inputs['loss_scale'], torch.ones(1, 2))
