# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest

from twinkle.loss import CrossEntropyLoss, GRPOLoss
from twinkle.loss.base import Loss
from twinkle.model.micro_batch import MicroBatchConfig, plan_micro_batches, select_batch, sequence_length
from twinkle.model.optimizer_group import BaseOptimizerGroup
from twinkle.model.transformers.transformers import TransformersModel
from twinkle.processor import InputProcessor


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


def test_loss_without_micro_batch_semantics_fails_when_split():
    with pytest.raises(NotImplementedError, match='does not support micro-batching'):
        Loss().micro_batch_scale([{}, {}], [0])


def test_transformers_forward_backward_keeps_original_default_path():
    class ModelHarness:
        _forward_backward_single = TransformersModel._forward_backward_single
        _not_encoded = staticmethod(lambda inputs: False)

        def _get_default_group(self):
            return 'adapter'

        def __init__(self):
            self.calls = []
            self.optimizer_group = {'adapter': BaseOptimizerGroup(loss_instance=CrossEntropyLoss())}

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
    class OptimizerConfig(BaseOptimizerGroup):
        def __init__(self):
            super().__init__()
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
        _forward_backward_single = TransformersModel._forward_backward_single
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

    assert outputs['loss'] == 2.0
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


@pytest.mark.parametrize('algorithm', [None, 'ffd', 'kk'])
def test_forward_only_micro_batches_restore_order_and_slice_masks(algorithm):
    import torch

    class ModelHarness:
        _build_micro_batch_plan = TransformersModel._build_micro_batch_plan

        def __init__(self):
            self.optimizer_group = {'adapter': SimpleNamespace(
                processor=InputProcessor(padding_free=False), _dp_group=None,
                _ensure_dp_group=lambda: None,
            )}
            self.calls = []

        def _get_default_group(self):
            return 'adapter'

        @staticmethod
        def _not_encoded(_inputs):
            return False

        _prepare_forward_only_inputs = TransformersModel._prepare_forward_only_inputs
        _forward_only_micro_batches = TransformersModel._forward_only_micro_batches

        def _forward_only_single(self, *, inputs, sampling_masks, **kwargs):
            ids = [item['sample_id'] for item in inputs]
            self.calls.append(ids)
            assert sampling_masks.tolist() == ids
            assert kwargs['disable_lora'] is True
            assert kwargs['task'] == 'embedding'
            width = max(len(item['input_ids']) for item in inputs)
            return {'logps': torch.tensor(ids)[:, None].expand(-1, width),
                    'logits': None, 'routed_experts': torch.tensor(ids)[:, None, None, None]}

    inputs = [{'input_ids': [1] * length, 'sample_id': index}
              for index, length in enumerate([2, 5, 1, 4, 3])]
    model = ModelHarness()
    options = {'micro_batch_size': 2}
    if algorithm is not None:
        options.update(dynamic_batching=True, max_tokens_per_micro_batch=10, packing_algorithm=algorithm)
    outputs = TransformersModel.forward_only.__wrapped__(
        model, inputs=inputs, sampling_masks=torch.arange(5), disable_lora=True, task='embedding', **options)
    assert len(model.calls) >= 3
    assert outputs['logps'].shape == (5, 5)
    assert outputs['logps'][:, 0].tolist() == list(range(5))
    assert outputs['routed_experts'][:, 0, 0, 0].tolist() == list(range(5))
    assert outputs['logits'] is None


def test_forward_only_without_micro_batches_prepares_inputs_once():
    class ModelHarness:
        optimizer_group = {'adapter': object()}

        def _get_default_group(self):
            return 'adapter'

        def _prepare_forward_only_inputs(self, inputs, optimizer_config):
            assert optimizer_config is self.optimizer_group['adapter']
            self.prepare_calls += 1
            return [inputs]

        def _forward_only_single(self, *, inputs, optimizer_config, **kwargs):
            assert optimizer_config is self.optimizer_group['adapter']
            return inputs, kwargs

    model = ModelHarness()
    model.prepare_calls = 0
    inputs = {'input_ids': [1]}
    result = TransformersModel.forward_only.__wrapped__(model, inputs=inputs, disable_lora=True)
    assert result == ([inputs], {'disable_lora': True})
    assert model.prepare_calls == 1


def test_merge_micro_batch_outputs_handles_nested_results():
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    result = merge_micro_batch_outputs([
        {'hidden_states': (torch.tensor([[2., 2.], [0., 0.]]),)},
        {'hidden_states': (torch.tensor([[1., 1.]]),)},
    ], [[2, 0], [1]])
    assert result['hidden_states'][0][:, 0].tolist() == [0., 1., 2.]


@pytest.mark.parametrize('padding_free', [False, True])
@pytest.mark.parametrize('algorithm', [None, 'ffd', 'kk'])
def test_forward_only_tiny_model_matches_unsplit_outputs(padding_free, algorithm):
    import torch
    from unittest.mock import patch
    from tests.model.test_value_model import _tiny_model_dir

    model_dir = _tiny_model_dir()
    if padding_free:
        # GPT2 has no packed attention backend. Remove attention layers so this
        # test isolates packing/unpacking without cross-document attention leakage.
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config.from_pretrained(model_dir)
        config.n_layer = 0
        GPT2LMHeadModel(config).save_pretrained(model_dir)
    model = TransformersModel(model_id=model_dir, mixed_precision='no')
    model.set_loss(CrossEntropyLoss())
    model.set_processor(InputProcessor(padding_free=padding_free))
    model._lazy_wrap_model()
    device = str(next(model.model.parameters()).device)
    inputs = [{'input_ids': [1, 2, 3, 4], 'labels': [-100, 2, 3, 4]},
              {'input_ids': [1, 2], 'labels': [-100, 2]},
              {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]}]
    if padding_free:
        for sample in inputs:
            sample['position_ids'] = list(range(len(sample['input_ids'])))
    options = {'micro_batch_size': 2}
    if algorithm:
        options.update(dynamic_batching=True, max_tokens_per_micro_batch=8, packing_algorithm=algorithm)
    with patch('twinkle.processor.base.Platform.get_local_device', return_value=device):
        expected = model.forward_only(inputs=inputs, return_logits=True)
        actual = model.forward_only(inputs=inputs, return_logits=True, **options)
    for index, sample in enumerate(inputs):
        length = len(sample['input_ids'])
        torch.testing.assert_close(actual['logps'][index, :length], expected['logps'][index, :length])
        torch.testing.assert_close(actual['logits'][index, :length], expected['logits'][index, :length])
    assert not actual['logps'].requires_grad


def test_forward_only_micro_batches_match_value_model():
    import torch
    from unittest.mock import patch
    from tests.model.test_value_model import _tiny_model_dir
    from twinkle.loss import PPOValueLoss
    from twinkle.model import TransformersValueModel

    model = TransformersValueModel(model_id=_tiny_model_dir(), mixed_precision='no')
    model.set_loss(PPOValueLoss())
    model._lazy_wrap_model()
    device = str(next(model.model.parameters()).device)
    inputs = [{'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]},
              {'input_ids': [1, 2], 'labels': [-100, 2]},
              {'input_ids': [1, 2, 3, 4], 'labels': [-100, 2, 3, 4]}]
    with patch('twinkle.processor.base.Platform.get_local_device', return_value=device):
        expected = model.forward_only(inputs=inputs)
        actual = model.forward_only(inputs=inputs, micro_batch_size=1)
    for index, sample in enumerate(inputs):
        length = len(sample['input_ids'])
        torch.testing.assert_close(actual['values'][index, :length], expected['values'][index, :length])
    assert actual.get('logps') is None
    assert not actual['values'].requires_grad


@pytest.mark.parametrize('reduction, expected', [('mean', 3.5), ('sum', 6.0)])
def test_forward_only_accumulates_aux_loss_using_training_micro_batch_scale(reduction, expected):
    import torch

    loss = CrossEntropyLoss(reduction=reduction)

    class ModelHarness:
        optimizer_group = {'adapter': SimpleNamespace(get_loss=lambda key: loss)}

        def _get_default_group(self):
            return 'adapter'

        def _not_encoded(self, inputs):
            return False

        def _build_micro_batch_plan(self, inputs, config, optimizer_config):
            return [[0], [1]]

        _prepare_forward_only_inputs = TransformersModel._prepare_forward_only_inputs
        _forward_only_micro_batches = TransformersModel._forward_only_micro_batches

        def _forward_only_single(self, *, inputs, **kwargs):
            value = float(inputs[0]['input_ids'][0])
            return {'aux_loss': torch.tensor(value), 'logps': torch.tensor([[value]])}

    inputs = [{'input_ids': [2], 'labels': [1]},
              {'input_ids': [4, 4, 4], 'labels': [1, 1, 1]}]
    outputs = TransformersModel.forward_only.__wrapped__(ModelHarness(), inputs=inputs, micro_batch_size=1)
    assert outputs['aux_loss'].item() == expected
    assert outputs['logps'][:, 0].tolist() == [2., 4.]


@pytest.mark.parametrize('plan', [[[0], [0]], [[0], [2]], [[0], []]])
def test_merge_rejects_invalid_sample_plan(plan):
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    with pytest.raises(ValueError, match='plan'):
        merge_micro_batch_outputs([{'logps': torch.ones(1, 2)}] * 2, plan)


@pytest.mark.parametrize('field, task, first_shape, second_shape', [
    ('logits', 'causal_lm', (1, 2, 4), (1, 3, 5)),
    ('logits', 'seq_cls', (1, 2), (1, 3)),
    ('embeddings', 'embedding', (1, 2), (1, 3)),
    ('routed_experts', 'causal_lm', (1, 2, 3, 2), (1, 4, 3, 1)),
])
def test_merge_rejects_non_sequence_dimension_changes(field, task, first_shape, second_shape):
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    with pytest.raises(ValueError, match=f'outputs.{field}: non-sequence dimension'):
        merge_micro_batch_outputs([{field: torch.zeros(first_shape)}, {field: torch.zeros(second_shape)}],
                                  [[0], [1]], task=task)


def test_merge_pooled_outputs_and_unknown_nested_outputs():
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    result = merge_micro_batch_outputs([
        {'embeddings': torch.tensor([[2., 3.]]), 'extra': {'scores': torch.tensor([[20.]])}},
        {'embeddings': torch.tensor([[0., 1.]]), 'extra': {'scores': torch.tensor([[0., 1.]])}},
    ], [[1], [0]], task='embedding')
    assert result['embeddings'].tolist() == [[0., 1.], [2., 3.]]
    assert result['extra']['scores'].tolist() == [[0., 1.], [20., 0.]]


def test_merge_fallback_validates_tuple_structure_and_pads_tensors():
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    outputs = [{'extra_layers': (torch.ones(1, 2, 2, 2),)},
               {'extra_layers': (torch.ones(1, 2, 3, 3) * 2,)}]
    result = merge_micro_batch_outputs(outputs, [[1], [0]])
    assert result['extra_layers'][0].shape == (2, 2, 3, 3)
    assert result['extra_layers'][0][0, 0, 0, 0] == 2
    outputs[1]['extra_layers'] = ()
    with pytest.raises(ValueError, match='outputs.extra_layers: tuple lengths'):
        merge_micro_batch_outputs(outputs, [[1], [0]])


def test_merge_aux_loss_requires_explicit_training_scales():
    import torch
    from twinkle.model.micro_batch import merge_micro_batch_outputs

    outputs = [{'aux_loss': torch.tensor(2.), 'logps': torch.tensor([[20.]])},
               {'aux_loss': torch.tensor(4.), 'logps': torch.tensor([[10.]])}]
    with pytest.raises(ValueError, match='outputs.aux_loss: one training scale'):
        merge_micro_batch_outputs(outputs, [[1], [0]])
    result = merge_micro_batch_outputs(outputs, [[1], [0]], micro_batch_scales=[.25, .75])
    assert result['aux_loss'].item() == 3.5
    assert result['logps'][:, 0].tolist() == [10., 20.]


@pytest.mark.parametrize('micro_batch_size', [None, 1])
def test_forward_only_encodes_trajectories_once(micro_batch_size):
    import torch
    from unittest.mock import Mock

    encoded = [{'input_ids': [1, 2]}, {'input_ids': [3]}]
    template = SimpleNamespace(batch_encode=Mock(return_value=encoded))
    optimizer_config = SimpleNamespace(template=template)

    class ModelHarness:
        optimizer_group = {'adapter': optimizer_config}
        _prepare_forward_only_inputs = TransformersModel._prepare_forward_only_inputs
        _forward_only_micro_batches = TransformersModel._forward_only_micro_batches

        def _get_default_group(self):
            return 'adapter'

        def _not_encoded(self, inputs):
            return 'input_ids' not in inputs

        def _build_micro_batch_plan(self, inputs, config, optimizer_config):
            return [[0], [1]]

        def _forward_only_single(self, *, inputs, optimizer_config, **kwargs):
            assert optimizer_config.template is template
            assert all('input_ids' in item for item in inputs)
            return {'embeddings': torch.tensor([[item['input_ids'][0]] for item in inputs])}

    inputs = [{'messages': []}, {'messages': []}]
    options = {} if micro_batch_size is None else {'micro_batch_size': micro_batch_size}
    result = TransformersModel.forward_only.__wrapped__(ModelHarness(), inputs=inputs, **options)
    template.batch_encode.assert_called_once_with(inputs)
    assert result['embeddings'].tolist() == [[1], [3]]


class TestMicroBatchConfigValidation:

    def test_non_positive_micro_batch_size_is_rejected(self):
        import pytest
        with pytest.raises(ValueError, match='micro_batch_size must be positive'):
            MicroBatchConfig(micro_batch_size=0)

    def test_unknown_packing_algorithm_is_rejected(self):
        import pytest
        with pytest.raises(ValueError, match='packing_algorithm must be ffd or kk'):
            MicroBatchConfig(micro_batch_size=1, dynamic_batching=True, max_tokens_per_micro_batch=8,
                             packing_algorithm='binpack')

    @pytest.mark.parametrize('max_tokens', [None, 0, -1])
    def test_dynamic_batching_requires_positive_token_budget(self, max_tokens):
        with pytest.raises(ValueError, match='max_tokens_per_micro_batch must be positive'):
            MicroBatchConfig(micro_batch_size=1, dynamic_batching=True, max_tokens_per_micro_batch=max_tokens)

    def test_static_config_needs_no_token_budget(self):
        config = MicroBatchConfig(micro_batch_size=2, dynamic_batching=False, max_tokens_per_micro_batch=None)
        assert config.packing_algorithm == 'ffd'


class TestMicroBatchConfigFromKwargs:

    def test_absent_options_yield_none(self):
        assert MicroBatchConfig.from_kwargs({}) is None
        assert MicroBatchConfig.from_kwargs({'learning_rate': 0.1}) is None

    def test_partial_configuration_requires_micro_batch_size(self):
        with pytest.raises(ValueError, match='micro_batch_size is required'):
            MicroBatchConfig.from_kwargs({'dynamic_batching': True})

    def test_full_configuration_pops_every_option(self):
        kwargs = {
            'micro_batch_size': '2',
            'dynamic_batching': 1,
            'max_tokens_per_micro_batch': 64,
            'packing_algorithm': 'kk',
            'learning_rate': 0.1,
        }
        config = MicroBatchConfig.from_kwargs(kwargs)
        assert config == MicroBatchConfig(micro_batch_size=2, dynamic_batching=True,
                                          max_tokens_per_micro_batch=64, packing_algorithm='kk')
        # from_kwargs consumes only its own options; foreign keys stay untouched.
        assert kwargs == {'learning_rate': 0.1}


class TestSequenceLength:

    def test_tensor_input_uses_last_dimension(self):
        import torch
        assert sequence_length({'input_ids': torch.zeros(2, 5, dtype=torch.long)}) == 5

    def test_list_input_uses_length(self):
        assert sequence_length({'input_ids': [1, 2, 3]}) == 3


class TestSelectBatch:

    def test_list_with_matching_length_is_indexed(self):
        value = ['a', 'b', 'c']
        assert select_batch(value, [2, 0], 3) == ['c', 'a']

    def test_list_with_foreign_length_is_returned_untouched(self):
        value = ['a', 'b']
        result = select_batch(value, [0], 3)
        assert result == ['a', 'b']
        assert result is value

    def test_tuple_with_matching_length_is_indexed(self):
        value = ('a', 'b', 'c')
        assert select_batch(value, [1], 3) == ('b',)

    def test_tensor_with_matching_first_dim_is_indexed(self):
        import torch
        value = torch.arange(6).reshape(3, 2)
        assert torch.equal(select_batch(value, [2, 0], 3), value[[2, 0]])

    def test_values_without_shape_pass_through(self):
        assert select_batch({'k': 1}, [0], 3) == {'k': 1}
        assert select_batch(3.5, [0], 3) == 3.5
