# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest

from twinkle.loss import CrossEntropyLoss, GRPOLoss
from twinkle.loss.base import Loss
from twinkle.model.micro_batch import MicroBatchConfig, plan_micro_batches
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.utils.nccl_safe import safe_loss


@pytest.mark.parametrize('packing_algorithm', ['ffd', 'kk'])
def test_dynamic_micro_batch_plan_preserves_samples_and_limits_cost(packing_algorithm):
    lengths = [10, 9, 8, 7, 4, 3, 2, 1]
    inputs = [{'input_ids': list(range(length))} for length in lengths]
    config = MicroBatchConfig(
        micro_batch_size=3,
        dynamic_batching=True,
        max_tokens_per_micro_batch=18,
        packing_algorithm=packing_algorithm,
    )

    plan = plan_micro_batches(inputs, config, padding_free=False)

    assert sorted(index for batch in plan for index in batch) == list(range(len(inputs)))
    for batch in plan:
        assert len(batch) <= config.micro_batch_size
        assert max(lengths[index] for index in batch) * len(batch) <= 18


def test_padding_free_dynamic_batching_uses_unpadded_token_cost():
    lengths = [10, 8, 6, 4]
    inputs = [{'input_ids': list(range(length))} for length in lengths]
    config = MicroBatchConfig(
        micro_batch_size=3,
        dynamic_batching=True,
        max_tokens_per_micro_batch=18,
    )

    plan = plan_micro_batches(inputs, config, padding_free=True)

    assert sorted(index for batch in plan for index in batch) == list(range(len(inputs)))
    assert all(sum(lengths[index] for index in batch) <= 18 for batch in plan)


def test_token_mean_loss_weight_uses_valid_label_count():
    inputs = [
        {'labels': [1, 2, -100, -100]},
        {'labels': [3, 4, 5, -100]},
        {'labels': [6, -100, -100, -100]},
    ]

    loss = CrossEntropyLoss(reduction='mean')
    first_weight = loss.micro_batch_scale(inputs, [0, 2])
    second_weight = loss.micro_batch_scale(inputs, [1])

    assert first_weight == .5
    assert second_weight == .5


def test_sample_mean_and_token_sum_micro_batch_scales():
    inputs = [
        {'labels': [1, -100]},
        {'labels': [2, 3]},
        {'labels': [4, -100]},
        {'labels': [5, 6]},
    ]

    assert GRPOLoss().micro_batch_scale(inputs, [0]) == .25
    assert CrossEntropyLoss(reduction='sum').micro_batch_scale(inputs, [0]) == 1.0


def test_safe_loss_preserves_wrapped_micro_batch_scale():
    inputs = [
        {'labels': [1, -100]},
        {'labels': [2, 3]},
        {'labels': [4, -100]},
        {'labels': [5, 6]},
    ]

    assert safe_loss(GRPOLoss()).micro_batch_scale(inputs, [0, 2]) == .5


def test_loss_without_micro_batch_semantics_fails_when_split():
    with pytest.raises(NotImplementedError, match='does not support micro-batching'):
        Loss().micro_batch_scale([{}, {}], [0])


def test_transformers_forward_backward_keeps_original_default_path():
    class ModelHarness:
        def __init__(self):
            self.calls = []

        def forward(self, *, inputs, **_kwargs):
            self.calls.append(('forward', inputs))
            return {}

        def calculate_loss(self, **_kwargs):
            self.calls.append(('loss', None))
            return 2.0

        def backward(self, **_kwargs):
            self.calls.append(('backward', None))

    model = ModelHarness()
    outputs = TransformersModel.forward_backward.__wrapped__(
        model,
        inputs=[{'input_ids': [1, 2]}],
    )

    assert [name for name, _ in model.calls] == ['forward', 'loss', 'backward']
    assert outputs['loss'] == 2.0


def test_transformers_forward_backward_executes_real_micro_batches():
    class OptimizerConfig:
        def __init__(self):
            self.processor = InputProcessor(padding_free=False)
            self.template = None
            self.train_status = SimpleNamespace(loss_value=None, num_tokens=0.0)
            self.loss_instance = SimpleNamespace(
                micro_batch_scale=lambda inputs, indices: len(indices) / len(inputs),
            )
            self._dp_group = None

        def _ensure_dp_group(self):
            pass

    class ModelHarness:
        _build_micro_batch_plan = TransformersModel._build_micro_batch_plan
        _forward_backward_micro_batch = TransformersModel._forward_backward_micro_batch
        _forward_backward_micro_batches = TransformersModel._forward_backward_micro_batches

        def __init__(self):
            self.optimizer_group = {'adapter': OptimizerConfig()}
            self.forward_batches = []
            self.backward_calls = []

        def _get_default_group(self):
            return 'adapter'

        @staticmethod
        def _not_encoded(_inputs):
            return False

        def forward(self, *, inputs, **_kwargs):
            self.forward_batches.append([item['sample_id'] for item in inputs])
            return {}

        def calculate_loss(self, **_kwargs):
            self.optimizer_group['adapter'].train_status.loss_value = 2.0
            self.optimizer_group['adapter'].train_status.num_tokens += 1.0
            return 2.0

        def backward(self, *, sync_gradients, **_kwargs):
            loss = self.optimizer_group['adapter'].train_status.loss_value
            self.backward_calls.append((sync_gradients, loss))
            self.optimizer_group['adapter'].train_status.loss_value = None

    model = ModelHarness()
    inputs = [
        {
            'sample_id': index,
            'input_ids': list(range(index + 1)),
        }
        for index in range(4)
    ]

    outputs = TransformersModel.forward_backward.__wrapped__(
        model,
        inputs=inputs,
        adapter_name='adapter',
        micro_batch_size=2,
        sync_gradients=True,
    )

    assert model.forward_batches == [[0, 1], [2, 3]]
    assert model.backward_calls == [(False, 1.0), (True, 1.0)]
    assert model.optimizer_group['adapter'].train_status.num_tokens == 1.0


def test_fixed_micro_batch_plan_can_match_a_larger_dp_micro_batch_count():
    inputs = [{'input_ids': [index]} for index in range(4)]

    plan = plan_micro_batches(
        inputs,
        MicroBatchConfig(micro_batch_size=2),
        padding_free=False,
        min_micro_batches=3,
    )

    assert plan == [[0, 1], [2], [3]]


def test_dp_micro_batch_planning_propagates_remote_rank_error(monkeypatch):
    from twinkle.model.transformers import transformers as module

    class OptimizerConfig:
        processor = InputProcessor(padding_free=False)
        _dp_group = object()

        @staticmethod
        def _ensure_dp_group():
            pass

    def all_gather(states, local_state, *, group):
        assert group is OptimizerConfig._dp_group
        states[:] = [
            local_state,
            {
                'micro_batch_count': None,
                'input_count': 2,
                'error': 'ValueError: sequence length 20 exceeds the limit',
            },
        ]

    monkeypatch.setattr(module.dist, 'get_world_size', lambda _group: 2)
    monkeypatch.setattr(module.dist, 'all_gather_object', all_gather)

    with pytest.raises(RuntimeError, match='rank 1.*sequence length 20'):
        TransformersModel._build_micro_batch_plan(
            object(),
            [{'input_ids': [1]}, {'input_ids': [2]}],
            MicroBatchConfig(micro_batch_size=1),
            OptimizerConfig(),
        )

def test_dp_micro_batch_planning_rejects_common_count_on_all_ranks(monkeypatch):
    from twinkle.model.transformers import transformers as module

    class OptimizerConfig:
        processor = InputProcessor(padding_free=False)
        _dp_group = object()

        @staticmethod
        def _ensure_dp_group():
            pass

    def all_gather(states, local_state, *, group):
        assert group is OptimizerConfig._dp_group
        states[:] = [
            local_state,
            {
                'micro_batch_count': 3,
                'input_count': 3,
                'error': None,
            },
        ]

    monkeypatch.setattr(module.dist, 'get_world_size', lambda _group: 2)
    monkeypatch.setattr(module.dist, 'all_gather_object', all_gather)

    with pytest.raises(ValueError, match='same number of non-empty micro-batches'):
        TransformersModel._build_micro_batch_plan(
            object(),
            [{'input_ids': [1]}, {'input_ids': [2]}],
            MicroBatchConfig(micro_batch_size=1),
            OptimizerConfig(),
        )
