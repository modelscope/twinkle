"""Input-rule checks without model training or accelerator allocation."""
import pytest
import torch

from twinkle.model.transformers.strategy.sequence_parallel import DataKeySpec, SequenceParallelStrategy
from twinkle.data_format.sequence_fields import partition_model_inputs
from twinkle.processor.fields import resolve_key_specs
from twinkle.model.transformers.strategy.sequence_parallel.data_keys import SPKeyCollection, SPKeyValue


def _strategy(rank=0):
    strategy = SequenceParallelStrategy(sp_config={'ulysses_size': 2})
    ctx = strategy._ctx
    ctx.world_size = ctx.sp_world_size = 2
    ctx.rp_world_size = 1
    ctx._sp_rank = rank
    return strategy


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('length', [8, 9])
def test_builtin_labels_match_existing_pad_split(rank, length):
    strategy = _strategy(rank)
    positions = torch.arange(length).expand(2, -1)
    labels = torch.arange(2 * length).reshape(2, length)
    spec = strategy._ctx.input_key_registry.get('labels')
    padded_labels = strategy._ctx._pad_key(labels, spec, positions)
    padded_positions = strategy._ctx.pad(positions, padding_value=-1, position_ids=positions)
    old = strategy._ctx._split_key(padded_labels, spec, padded_positions)
    inputs = {'labels': labels.clone(), 'position_ids': positions}
    actual = strategy.preprocess_inputs(inputs)['labels']
    torch.testing.assert_close(actual, old, rtol=0, atol=0)
    torch.testing.assert_close(labels, torch.arange(2 * length).reshape(2, length))


@pytest.mark.parametrize('rank', [0, 1])
def test_custom_key_padding_dimension_roll_and_model_isolation(rank):
    strategy = _strategy(rank)
    other = _strategy(rank)
    spec = DataKeySpec('token_weights', split_dim=1, pad_value=-2, roll=True, pad_at='prepare', split_at='prepare')
    strategy.register_input_key(spec)
    strategy.register_input_key(spec)  # Identical declarations are safe.
    values = torch.arange(10).reshape(1, 5, 2)
    positions = torch.arange(5).unsqueeze(0)
    prepared = strategy.preprocess_inputs({'token_weights': values, 'position_ids': positions})
    expected = torch.cat((values, torch.full((1, 1, 2), -2)), dim=1).roll(-1, dims=1)
    torch.testing.assert_close(prepared['token_weights'], expected[:, rank * 3:(rank + 1) * 3])
    untouched = other.preprocess_inputs({'token_weights': values, 'position_ids': positions})
    assert untouched['token_weights'] is values


@pytest.mark.parametrize('rank', [0, 1])
def test_key_collection_executes_only_at_declared_boundary(rank):
    strategy = _strategy(rank)
    ctx = strategy._ctx
    ctx.tokenizer = type('Tokenizer', (), {'pad_token_id': 0})()
    token_type_spec = DataKeySpec(
        'token_type_ids',
        split_dim=1,
        pad_value=-1,
        stage='forward',
        pad_at='forward',
        split_at='forward',
    )
    strategy.register_input_key(token_type_spec)
    values = {
        'input_ids': torch.arange(5).expand(2, -1),
        'position_ids': torch.arange(5).expand(2, -1),
        'attention_mask': torch.ones((2, 5), dtype=torch.long),
        'token_type_ids': torch.arange(5).expand(2, -1),
    }
    specs = tuple(spec for spec in ctx.input_key_registry if spec.stage == 'forward')
    keys = SPKeyCollection.from_mapping(values, specs, common_names=('input_ids', 'position_ids', 'attention_mask'))
    ctx.pad_and_split_inputs(keys, boundary='forward', real_position_ids=values['position_ids'])
    assert keys.get('input_ids').value.shape == (2, 3)
    assert keys.get('position_ids').value.shape == (2, 3)
    assert keys.get('token_type_ids').value.shape == (2, 3)
    # attention_mask follows its own rule: pad to the global frame but do not shard.
    torch.testing.assert_close(keys.get('attention_mask').value, torch.tensor([[1, 1, 1, 1, 1, 0]]).expand(2, -1))


def test_key_collection_rejects_duplicate_names_across_common_and_extra():
    spec = DataKeySpec('token_type_ids', stage='forward', pad_at='forward', split_at='forward')
    keys = SPKeyCollection()
    keys.add(SPKeyValue(spec, torch.ones(1, 2)), common=True)
    with pytest.raises(ValueError, match='Duplicate SP key'):
        keys.add(SPKeyValue(spec, torch.zeros(1, 2)))


@pytest.mark.parametrize('rank', [0, 1])
def test_prepare_keeps_loss_scale_and_completion_mask_global(rank):
    strategy = _strategy(rank)
    positions = torch.arange(5).unsqueeze(0)
    loss_scale = torch.arange(1, 6, dtype=torch.float).unsqueeze(0)
    completion_mask = torch.tensor([[0, 0, 1, 1, 1]])
    inputs = {
        'position_ids': positions,
        'labels': torch.arange(5).unsqueeze(0),
        'loss_scale': loss_scale.clone(),
        'completion_mask': completion_mask.clone(),
    }
    prepared = strategy.preprocess_inputs(inputs)
    assert prepared['labels'].shape == (1, 3)
    torch.testing.assert_close(prepared['loss_scale'], loss_scale)
    torch.testing.assert_close(prepared['completion_mask'], completion_mask)


def test_invalid_rules_and_non_tensor_payload_fail_clearly():
    strategy = _strategy()
    with pytest.raises(ValueError, match='Invalid SP key stage'):
        DataKeySpec('cache_position', stage='passthrough')
    with pytest.raises(ValueError, match='Conflicting'):
        strategy.register_input_key(DataKeySpec('labels', pad_value=0))
    with pytest.raises(ValueError, match='forward hook'):
        strategy.register_input_key(DataKeySpec('input_ids'))
    with pytest.raises(ValueError, match='not an SP data key'):
        strategy.register_input_key(DataKeySpec('padding_free'))
    with pytest.raises(TypeError, match='must be a tensor'):
        strategy.preprocess_inputs({'labels': [1, 2]})


def test_disabled_sp_and_absent_keys_are_unchanged():
    strategy = SequenceParallelStrategy(sp_config={'enabled': False, 'ulysses_size': 2})
    inputs = {'labels': torch.tensor([[1, 2]])}
    assert strategy.preprocess_inputs(inputs) is inputs
    assert _strategy().preprocess_inputs({'labels': None}) == {'labels': None}


@pytest.mark.parametrize('sp_rank', [0, 1])
@pytest.mark.parametrize('rp_rank', [0, 1])
def test_packed_ring_labels_preserve_existing_layout(sp_rank, rp_rank):
    strategy = _strategy(sp_rank)
    ctx = strategy._ctx
    ctx.world_size = 4
    ctx.rp_world_size = 2
    ctx._rp_rank = rp_rank
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]])
    labels = torch.arange(8).unsqueeze(0)
    spec = ctx.input_key_registry.get('labels')
    padded_labels = ctx._pad_key(labels, spec, positions)
    padded_positions = ctx.pad(positions, padding_value=-1, position_ids=positions)
    old = ctx._split_key(padded_labels, spec, padded_positions)
    new = strategy.preprocess_inputs({'labels': labels, 'position_ids': positions})['labels']
    torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_packed_unpack_uses_explicit_per_key_unpack_rule():
    from twinkle.processor.base import InputProcessor

    processor = InputProcessor(padding_free=True)
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    labels = torch.tensor([[10, 11, 12, 20, 21]])
    completion_mask = torch.tensor([[0, 1, 1, 0, 1]])
    loss_scale = torch.tensor([[1., 2., 3., 4., 5.]])
    inputs = {
        'position_ids': positions,
        'labels': labels,
        'completion_mask': completion_mask,
        'loss_scale': loss_scale,
    }
    unpacked, _ = processor.unpack_packed_sequences(inputs, outputs=None)
    torch.testing.assert_close(unpacked['labels'], torch.tensor([[10, 11, 12], [20, 21, -100]]))
    torch.testing.assert_close(unpacked['completion_mask'], torch.tensor([[0, 1, 1], [0, 1, 0]]))
    torch.testing.assert_close(unpacked['loss_scale'], torch.tensor([[1., 2., 3.], [4., 5., 0.]]))


@pytest.mark.parametrize('rank', [0, 1])
def test_processor_preserves_registered_keys_before_sp_split(monkeypatch, rank):
    from twinkle import Platform
    from twinkle.processor.base import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    strategy = _strategy(rank)
    strategy.register_input_key(DataKeySpec('token_weights', pad_value=-2, pad_at='prepare', split_at='prepare'))
    processor = InputProcessor()
    values = torch.arange(5).unsqueeze(0)
    sample = {'input_ids': torch.ones(1, 5, dtype=torch.long),
              'position_ids': torch.arange(5).unsqueeze(0),
              'labels': torch.ones(1, 5, dtype=torch.long),
              'token_weights': values, 'unregistered': values}
    result = processor([sample], input_key_specs=strategy.input_key_specs)
    result = strategy.preprocess_inputs(result)
    expected = torch.tensor([[0, 1, 2, 3, 4, -2]])[:, rank * 3:(rank + 1) * 3]
    torch.testing.assert_close(result['token_weights'], expected)
    assert 'unregistered' not in result
    other = _strategy(rank)
    assert 'token_weights' not in processor.to_transformers_dict([sample], input_key_specs=other.input_key_specs)[0]


@pytest.mark.parametrize('rank', [0, 1])
def test_builtin_completion_mask_and_loss_scale_stay_global_during_prepare(rank):
    strategy = _strategy(rank)
    payload = {'position_ids': torch.arange(5).unsqueeze(0),
               'completion_mask': torch.tensor([[0, 0, 1, 1, 1]]),
               'loss_scale': torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])}
    prepared = strategy.preprocess_inputs(payload)
    torch.testing.assert_close(prepared['completion_mask'], torch.tensor([[0, 0, 1, 1, 1]]))
    torch.testing.assert_close(prepared['loss_scale'], torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))


def test_registered_multidimensional_key_collation_and_loss_boundary(monkeypatch):
    from twinkle import Platform
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    strategy = _strategy()
    strategy.register_input_key(DataKeySpec(
        'token_features',
        pad_value=-2,
        gather_at='loss',
        pad_at='prepare',
        split_at='prepare',
    ))
    processor = InputProcessor()
    samples = []
    for length in (3, 5):
        samples.append({'input_ids': list(range(1, length + 1)),
                        'position_ids': list(range(length)), 'labels': list(range(1, length + 1)),
                        'token_features': [ [[1] * length, [2] * length] ],
                        'completion_mask': [1] * length})
    result = processor(samples, input_key_specs=strategy.input_key_specs)
    result = strategy.preprocess_inputs(result)
    expected = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]])
    torch.testing.assert_close(result['token_features'], expected)
    result, auxiliary = partition_model_inputs(result, strategy.input_key_specs)
    assert set(auxiliary) == {'labels', 'completion_mask', 'token_features'}
    assert 'token_features' not in result


def test_model_specific_forward_key_is_split_only_in_model_hook(monkeypatch):
    from types import SimpleNamespace
    from twinkle import Platform
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    strategy = _strategy(1)
    strategy._ctx.tokenizer = SimpleNamespace(pad_token_id=0)
    strategy.register_input_key(DataKeySpec('token_type_ids', stage='forward', pad_at='forward', split_at='forward'))

    class Model(torch.nn.Module):
        def forward(self, token_type_ids, **kwargs):
            return token_type_ids

    model = Model()
    strategy._ctx._prepare_forward_hook(model)
    ids = torch.arange(5).unsqueeze(0)
    prepared = InputProcessor()([{'input_ids': ids, 'position_ids': ids,
                                  'token_type_ids': ids}], input_key_specs=strategy.input_key_specs)
    prepared = strategy.preprocess_inputs(prepared)
    assert prepared['token_type_ids'].shape == (1, 5)
    torch.testing.assert_close(model(**prepared), torch.tensor([[3, 4, 0]]))


def test_declared_collection_uses_sequence_axis_and_collects_without_logps(monkeypatch):
    import twinkle.model.transformers.strategy.sequence_parallel as sp

    strategy = _strategy()
    strategy._ctx.extra_kwargs['position_ids'] = torch.arange(5).unsqueeze(0)
    strategy.register_input_key(DataKeySpec(
        'token_features',
        gather_at='loss',
        pad_value=-2,
        pad_at='prepare',
        split_at='prepare',
    ))
    strategy.register_output_key(DataKeySpec(
        'scores',
        split_dim=2,
        side='output',
        stage='output',
        gather_at='loss',
        autograd=True,
    ))
    monkeypatch.setattr(strategy._ctx, 'gather', lambda value, dim, position_ids: torch.cat((value, value), dim=dim))
    monkeypatch.setattr(sp.GatherLoss, 'apply', lambda ctx, value, labels, dim, pos: (ctx.gather(value, dim, pos), None))
    local = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    inputs, outputs = strategy.gather_loss_tensors({'token_features': local}, {'scores': local.float()})
    expected = torch.tensor([[[1, 2, 3, 1, 2], [4, 5, 6, 4, 5]]])
    torch.testing.assert_close(inputs['token_features'], expected)
    torch.testing.assert_close(outputs['scores'], expected.float())


def test_packing_unpacks_registered_keys_on_declared_axis():
    from twinkle.processor import InputProcessor
    from twinkle.model.transformers.strategy.sequence_parallel.data_keys import BUILTIN_INPUT_KEYS

    input_specs = (*BUILTIN_INPUT_KEYS, DataKeySpec(
        'token_features',
        pad_value=-2,
        gather_at='loss',
        pad_at='prepare',
        split_at='prepare',
    ))
    output_specs = (DataKeySpec('scores', side='output', stage='output', pad_value=-3),)
    inputs = {'position_ids': torch.tensor([[0, 1, 2, 0, 1]]), 'labels': torch.tensor([[1, 2, 3, 4, 5]]),
              'completion_mask': torch.ones(1, 5), 'token_features': torch.tensor([[[1, 2, 3, 4, 5]]])}
    outputs = {'scores': torch.tensor([[[10, 20, 30, 40, 50]]])}
    unpacked, unpacked_outputs = InputProcessor().unpack_packed_sequences(
        inputs, outputs, input_key_specs=input_specs, output_key_specs=output_specs)
    assert unpacked['token_features'].shape == (2, 1, 3)
    assert unpacked['token_features'].tolist() == [[[1, 2, 3]], [[4, 5, -2]]]
    assert unpacked['completion_mask'].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert unpacked_outputs['scores'].tolist() == [[[10, 20, 30]], [[40, 50, -3]]]


@pytest.mark.parametrize('padding_free', [False, True])
def test_processor_preserves_custom_keys_without_sp_strategy(monkeypatch, padding_free):
    from twinkle import Platform
    from twinkle.processor import InputProcessor
    from twinkle.model.transformers.strategy.sequence_parallel.data_keys import BUILTIN_INPUT_KEYS

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    specs = (*BUILTIN_INPUT_KEYS, DataKeySpec('token_features', pad_value=-2))
    samples = [
        {'input_ids': list(range(length)), 'position_ids': list(range(length)),
         'token_features': [[[1] * length, [2] * length]]}
        for length in (3, 5)
    ]
    processor = InputProcessor(padding_free=padding_free)
    result = processor(samples, input_key_specs=specs)
    expected = (torch.tensor([[[1] * 8, [2] * 8]]) if padding_free else
                torch.tensor([[[1, 1, 1, -2, -2], [2, 2, 2, -2, -2]], [[1] * 5, [2] * 5]]))
    torch.testing.assert_close(result['token_features'], expected)
    result, auxiliary = partition_model_inputs(result, specs)
    torch.testing.assert_close(auxiliary['token_features'], expected)


@pytest.mark.parametrize('padding_free', [False, True])
@pytest.mark.parametrize('micro_batch_size,variable_seq_lengths', [(None, False), (2, False), (1, False), (1, True)])
def test_collation_preserves_declared_sequence_axis(padding_free, micro_batch_size, variable_seq_lengths):
    from twinkle.processor import InputProcessor
    from twinkle.model.transformers.strategy.sequence_parallel.data_keys import BUILTIN_INPUT_KEYS

    specs = (*BUILTIN_INPUT_KEYS, DataKeySpec('token_features', split_dim=-1, pad_value=-2))
    processor = InputProcessor(padding_free=padding_free)
    samples = [
        {'input_ids': torch.tensor([[1, 2, 3]]),
         'position_ids': torch.tensor([[0, 1, 2]]),
         'token_features': torch.tensor([[[10, 11, 12], [20, 21, 22]]])},
        {'input_ids': torch.tensor([[4, 5, 6, 7, 8]]),
         'position_ids': torch.tensor([[0, 1, 2, 3, 4]]),
         'token_features': torch.tensor([[[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]]])},
    ]

    results = processor.collate_fn(samples, micro_batch_size=micro_batch_size,
                                   variable_seq_lengths=variable_seq_lengths, input_key_specs=specs)
    first = torch.tensor([[[10, 11, 12], [20, 21, 22]]])
    first_padded = torch.tensor([[[10, 11, 12, -2, -2], [20, 21, 22, -2, -2]]])
    second = torch.tensor([[[30, 31, 32, 33, 34], [40, 41, 42, 43, 44]]])
    if micro_batch_size == 1:
        expected = [first if padding_free or variable_seq_lengths else first_padded, second]
    elif padding_free:
        expected = [torch.cat((first, second), dim=-1)]
    else:
        expected = [torch.cat((first_padded, second), dim=0)]
    assert len(results) == len(expected)
    for result, value in zip(results, expected):
        torch.testing.assert_close(result['token_features'], value)


def test_processor_preserves_special_arguments_outside_data_key_registry():
    from twinkle.processor import InputProcessor

    strategy = _strategy()
    spec_names = {spec.name for spec in strategy.input_key_specs}
    assert 'cache_position' not in spec_names
    assert 'padding_free' not in spec_names

    sample = {'input_ids': torch.tensor([[1, 2]]),
              'cache_position': torch.tensor([0, 1]),
              'padding_free': True}
    converted_with_sp = InputProcessor.to_transformers_dict([sample], input_key_specs=strategy.input_key_specs)[0]
    converted_without_sp = InputProcessor.to_transformers_dict([sample])[0]
    for converted in (converted_with_sp, converted_without_sp):
        torch.testing.assert_close(converted['cache_position'], sample['cache_position'])
        assert converted['padding_free'] is True


@pytest.mark.parametrize('padding_free', [False, True])
def test_common_loss_fields_collate_and_unpack_without_sp(monkeypatch, padding_free):
    from types import SimpleNamespace
    from twinkle import Platform
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    loss = SimpleNamespace(input_key_specs=(DataKeySpec('token_targets', split_dim=2, pad_value=-9),))
    input_specs, output_specs = resolve_key_specs(loss)
    spec = next(spec for spec in input_specs if spec.name == 'token_targets')
    assert not spec.need_pad and not spec.need_split and not spec.gather_at
    samples = [dict(input_ids=list(range(length)), position_ids=list(range(length)),
                    labels=list(range(length)), token_targets=[[[1] * length, [2] * length]])
               for length in (3, 5)]
    processor = InputProcessor(padding_free=padding_free)
    batch = processor(samples, input_key_specs=input_specs)
    model_inputs, loss_inputs = partition_model_inputs(batch, input_specs)
    assert 'token_targets' not in model_inputs and 'labels' not in model_inputs
    assert loss_inputs['token_targets'].shape == ((1, 2, 8) if padding_free else (2, 2, 5))
    restored, _ = processor.unpack_packed_sequences(
        batch, task='causal_lm', input_key_specs=input_specs, output_key_specs=output_specs)
    torch.testing.assert_close(restored['token_targets'], torch.tensor([
        [[1, 1, 1, -9, -9], [2, 2, 2, -9, -9]], [[1] * 5, [2] * 5]]))


@pytest.mark.parametrize('ring_size', [1, 2])
def test_global_loss_fields_are_never_gathered(monkeypatch, ring_size):
    strategy = _strategy()
    strategy._ctx.rp_world_size = ring_size
    strategy._ctx.sp_world_size = 2 // ring_size
    strategy._ctx.extra_kwargs['position_ids'] = torch.tensor([[0, 1, 2, 0, 1]])

    def forbidden(*args, **kwargs):
        raise AssertionError('Global loss fields must not enter SP collectives')

    monkeypatch.setattr(strategy._ctx, 'gather', forbidden)
    mask = torch.tensor([[0, 1, 1, 0, 1]])
    scale = torch.tensor([[0., 1., 1., 0., 1.]])
    inputs, _ = strategy.gather_loss_tensors({'completion_mask': mask, 'loss_scale': scale}, {})
    assert inputs['completion_mask'] is mask and inputs['loss_scale'] is scale


def test_model_input_destination_is_independent_of_split_stage():
    spec = DataKeySpec('token_context', stage='auxiliary', model_input=True)
    model_inputs, loss_inputs = partition_model_inputs({'token_context': 1, 'channel': ['a']}, (spec,))
    assert model_inputs == {'token_context': 1}
    assert loss_inputs == {'channel': ['a']}



def test_explicit_parallel_boundaries_opt_in_without_changing_loss_field_defaults():
    global_field = DataKeySpec('targets')
    assert not global_field.need_pad and not global_field.need_split
    local_field = DataKeySpec('local_targets', split_at='prepare', pad_at='prepare')
    assert local_field.need_split and local_field.need_pad
    padded_field = DataKeySpec('padded_targets', pad_at='prepare')
    assert padded_field.need_pad and not padded_field.need_split


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('stage', ['forward', 'auxiliary'])
def test_prepare_then_forward_honors_separate_pad_and_split_boundaries(rank, stage):
    from types import SimpleNamespace

    strategy = _strategy(rank)
    strategy._ctx.tokenizer = SimpleNamespace(pad_token_id=0)
    spec = DataKeySpec(
        'token_type_ids',
        split_dim=1,
        pad_value=-7,
        stage=stage,
        model_input=True,
        pad_at='prepare',
        split_at='forward',
    )
    strategy.register_input_key(spec)

    class Model(torch.nn.Module):
        def forward(self, token_type_ids, **kwargs):
            return token_type_ids

    model = Model()
    strategy._ctx._prepare_forward_hook(model)
    positions = torch.arange(5).unsqueeze(0)
    prepared = strategy.preprocess_inputs(dict(input_ids=positions.clone(), position_ids=positions,
                                               token_type_ids=positions.clone()))
    torch.testing.assert_close(prepared['token_type_ids'], torch.tensor([[0, 1, 2, 3, 4, -7]]))
    expected = torch.tensor([[0, 1, 2]]) if rank == 0 else torch.tensor([[3, 4, -7]])
    torch.testing.assert_close(model(**prepared), expected)


def test_direct_registration_survives_model_schema_resolution_and_processor(monkeypatch):
    from types import SimpleNamespace
    from twinkle import Platform
    from twinkle.model.transformers.transformers import TransformersModel
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    strategy = _strategy()
    spec = DataKeySpec('token_weights', pad_value=-2)
    strategy.register_input_key(spec)
    output_spec = DataKeySpec('token_scores', side='output', stage='output', gather_at='loss')
    strategy.register_output_key(output_spec)
    model = torch.nn.Linear(2, 2)
    wrapper = SimpleNamespace(model=model, strategy=SimpleNamespace(unwrap_model=lambda value: value),
                              sp_strategy=strategy)
    input_specs, output_specs = TransformersModel._get_field_specs(wrapper, None)
    assert spec in input_specs and output_spec in output_specs
    processor = InputProcessor()
    result = processor([dict(input_ids=[1, 2, 3], token_weights=[0.1, 0.2, 0.3])],
                       input_key_specs=input_specs)
    torch.testing.assert_close(result['token_weights'], torch.tensor([[0.1, 0.2, 0.3]]))
    _, loss_inputs = partition_model_inputs(result, input_specs)
    assert 'token_weights' in loss_inputs
    # Consuming provider schemas must not turn them into direct registrations.
    strategy.register_key_specs((DataKeySpec('provider_targets'),), ())
    assert strategy.field_declarations.input_key_specs == (spec,)


@pytest.mark.parametrize('label_type', ['tensor', 'numpy', 'python'])
@pytest.mark.parametrize('framework', ['transformers', 'megatron'])
def test_scalar_classification_labels_collate_with_common_schema(monkeypatch, label_type, framework):
    import numpy as np
    from twinkle import Platform
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    convert = {'tensor': torch.tensor, 'numpy': np.array, 'python': int}[label_type]
    batch = [dict(input_ids=[1, 2, 3], position_ids=[0, 1, 2], labels=convert(label)) for label in (1, 0)]
    result = InputProcessor(framework=framework)(batch)
    if framework == 'megatron':
        result = result[0]
    torch.testing.assert_close(result['labels'], torch.tensor([[1], [0]]))
    assert result['input_ids'].shape == (2, 3)


@pytest.mark.parametrize('with_labels', [False, True])
def test_packed_custom_loss_restoration_does_not_require_token_labels(with_labels):
    from types import SimpleNamespace
    from twinkle.processor import InputProcessor

    loss = SimpleNamespace(
        input_key_specs=(DataKeySpec('token_targets', pad_value=-9),),
        output_key_specs=(DataKeySpec('token_scores', side='output', stage='output'),))
    input_specs, output_specs = resolve_key_specs(loss)
    inputs = dict(position_ids=torch.tensor([[0, 1, 2, 0, 1]]), token_targets=torch.zeros(1, 5))
    if with_labels:
        inputs['labels'] = torch.full((1, 5), -100)
    outputs = {'token_scores': torch.tensor([[1., 1., 1., 3., 3.]])}
    restored, outputs = InputProcessor(padding_free=True).unpack_packed_sequences(
        inputs, outputs, input_key_specs=input_specs, output_key_specs=output_specs)
    targets, scores = restored['token_targets'], outputs['token_scores']
    torch.testing.assert_close(targets, torch.tensor([[0., 0., 0.], [0., 0., -9.]]))
    valid = targets != -9
    balanced_loss = (((scores - targets).square() * valid).sum(-1) / valid.sum(-1)).mean()
    torch.testing.assert_close(balanced_loss, torch.tensor(5.))
    assert ('labels' in restored) == with_labels


def test_packed_output_logits_restore_without_labels():
    from twinkle.processor import InputProcessor

    logits = torch.arange(10).reshape(1, 5, 2)
    _, outputs = InputProcessor(padding_free=True).unpack_packed_sequences(
        {'position_ids': torch.tensor([[0, 1, 2, 0, 1]])}, {'logits': logits})
    torch.testing.assert_close(outputs['logits'], torch.tensor([[[0, 1], [2, 3], [4, 5]],
                                                             [[6, 7], [8, 9], [0, 0]]]))


@pytest.mark.parametrize('operation', ['pad', 'split'])
def test_loss_only_fields_reject_unreachable_forward_operations(operation):
    from dataclasses import replace

    spec = (DataKeySpec('token_targets', pad_at='forward') if operation == 'pad' else
            DataKeySpec('token_targets', pad_at='prepare', split_at='forward'))
    with pytest.raises(ValueError, match='Loss-only input.*forward boundary'):
        _strategy().register_input_key(spec)
    # Model arguments can reach the forward hook, regardless of their stage name.
    _strategy().register_input_key(replace(spec, model_input=True))


@pytest.mark.parametrize('pad_at,split_at', [(None, None), ('prepare', None), (None, 'forward'),
                                             ('prepare', 'forward')])
def test_parallel_switches_are_derived_from_boundaries(pad_at, split_at):
    from dataclasses import fields

    spec = DataKeySpec('token_features', pad_at=pad_at, split_at=split_at, model_input=True)
    assert spec.need_pad is (pad_at is not None)
    assert spec.need_split is (split_at is not None)
    assert not {'need_pad', 'need_split', 'need_gather', 'owner'} & {field.name for field in fields(spec)}


@pytest.mark.parametrize('kwargs,message', [
    ({'pad_at': 'auxiliary'}, 'padding boundary'),
    ({'split_at': 'output'}, 'split boundary'),
    ({'gather_at': 'forward'}, 'gather boundary'),
    ({'gather_at': 'loss'}, 'Only sharded input'),
    ({'roll': True}, 'Rolling requires'),
    ({'side': 'output', 'stage': 'output', 'pad_at': 'prepare'}, 'Output keys cannot'),
])
def test_invalid_parallel_operation_boundaries_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DataKeySpec('token_features', **kwargs)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('explicit_mask', [False, True])
def test_single_sample_sdpa_builds_a_padded_global_attention_mask(rank, explicit_mask):
    from types import SimpleNamespace

    ctx = _strategy(rank)._ctx
    ctx.attn_implementation = 'sdpa'
    ctx.tokenizer = SimpleNamespace(pad_token_id=0)
    positions = torch.arange(5).unsqueeze(0)
    values = {'input_ids': positions.clone(), 'position_ids': positions,
              'attention_mask': torch.ones(1, 5, dtype=torch.long) if explicit_mask else None}
    specs = tuple(spec for spec in ctx.input_key_registry if spec.stage == 'forward')
    keys = SPKeyCollection.from_mapping(values, specs)
    ctx.pad_and_split_inputs(keys, boundary='forward', real_position_ids=positions)
    torch.testing.assert_close(keys.get_value('attention_mask'), torch.tensor([[1, 1, 1, 1, 1, 0]]))
    assert keys.get_value('input_ids').shape == (1, 3)


@pytest.mark.parametrize('configured,packed', [(False, False), (True, False), (False, True)])
def test_processor_public_layout_query(configured, packed):
    from twinkle.processor import InputProcessor

    positions = torch.tensor([[0, 1, 0, 1]]) if packed else torch.arange(4).unsqueeze(0)
    batch = {'position_ids': positions}
    processor = InputProcessor(padding_free=configured)
    assert processor.is_padding_free(batch) is (configured or packed)
    assert processor.is_padding_free([batch]) is (configured or packed)
    assert batch['position_ids'] is positions


def test_processor_supported_fields_include_multimodal_replay_and_control_inputs():
    from twinkle.processor import InputProcessor
    from twinkle.processor.fields import TRANSFORMERS_INPUT_KEYS, LAYOUT_ONLY_KEYS

    required = {'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw',
                'input_features', 'input_features_mask', 'feature_attention_mask', 'grid_thws',
                'routed_experts', 'replay_mask', 'mm_token_type_ids', 'second_per_grid_ts',
                'cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q', 'cu_seqlens_kv',
                'cache_position', 'padding_free', 'packed_seq_params'}
    assert required <= TRANSFORMERS_INPUT_KEYS
    assert set(InputProcessor.padding_map) <= TRANSFORMERS_INPUT_KEYS | set(LAYOUT_ONLY_KEYS)
    value = torch.ones(1, 2)
    sample = {key: value for key in required}
    sample.update(length=2, unregistered=value)
    output = InputProcessor.to_transformers_dict([sample])[0]
    assert set(output) == required
    assert all(output[key] is value for key in required)


def test_processor_defaults_do_not_schedule_parallel_operations():
    from twinkle.processor.fields import resolve_key_specs

    inputs, outputs = resolve_key_specs()
    assert all(spec.pad_at is None and spec.split_at is None and spec.gather_at is None
               for spec in (*inputs, *outputs))
    strategy = _strategy()
    strategy.register_key_specs(inputs, outputs)
    labels = strategy._ctx.input_key_registry.get('labels')
    assert labels.pad_at == labels.split_at == 'prepare'
    assert labels.gather_at == 'loss'
    assert strategy._ctx.output_key_registry.get('logps').gather_at == 'loss'


def test_direct_builtin_output_declaration_keeps_model_schema_and_sp_registry_consistent():
    from twinkle.processor.fields import resolve_key_specs

    strategy = _strategy()
    logits = strategy._ctx.output_key_registry.get('logits')
    strategy.register_output_key(logits)
    inputs, outputs = resolve_key_specs(strategy.field_declarations)
    assert next(spec for spec in outputs if spec.name == 'logits') == logits
    strategy.register_key_specs(inputs, outputs)


def _megatron_processor_cp_worker(rank, port):
    import os
    from datetime import timedelta
    from types import SimpleNamespace
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from twinkle.processor import InputProcessor
    from twinkle.loss import CrossEntropyLoss
    from twinkle.patch.megatron_emb import _output_embedding_hook

    os.environ['RANK'], os.environ['WORLD_SIZE'] = str(rank), '2'
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    original = mpu.get_context_parallel_group
    # Only resolve the CP group here; pad/split/gather and unpack are real code.
    mpu.get_context_parallel_group = lambda: dist.group.WORLD
    try:
        mesh = SimpleNamespace(cp_world_size=2, cp_rank=rank, tp_world_size=1, sequence_parallel=False)
        processor = InputProcessor(device_mesh=mesh, framework='megatron', padding_free=True)
        processor.device_mesh = mesh
        rows = [dict(input_ids=torch.arange(1, length + 1).unsqueeze(0),
                     position_ids=torch.arange(length).unsqueeze(0),
                     labels=torch.arange(1, length + 1).unsqueeze(0),
                     loss_scale=torch.ones(1, length), completion_mask=torch.ones(1, length, dtype=torch.long))
                for length in (3, 5)]
        padded = processor.pad_cp(rows)
        full = processor.collate_fn(padded)[0]
        full = processor.add_extra_padding_free_args([full])[0]
        cu = full['packed_seq_params'].cu_seqlens_q
        torch.testing.assert_close(cu, torch.tensor([0, 4, 12], dtype=cu.dtype))
        local = processor.split_cp([dict(full)])[0]
        reconstructed = dict(local)
        for key in ('labels', 'position_ids', 'loss_scale', 'completion_mask'):
            reconstructed[key] = processor.postprocess_tensor_cp(local[key], cu_seqlens=cu)
            torch.testing.assert_close(reconstructed[key], full[key])
        # Match Megatron.forward_step_func: CP reconstruction followed by unpack.
        full_logps = -full['labels'].clamp_min(0).float() / 10
        local_logps = -local['labels'].clamp_min(0).float() / 10
        gathered_logps = processor.postprocess_tensor_cp(local_logps, cu_seqlens=cu)
        expected_inputs, expected_outputs = processor.unpack_packed_sequences(full, {'logps': full_logps})
        actual_inputs, actual_outputs = processor.unpack_packed_sequences(reconstructed, {'logps': gathered_logps})
        for key in ('labels', 'completion_mask', 'loss_scale'):
            torch.testing.assert_close(actual_inputs[key], expected_inputs[key])
        torch.testing.assert_close(actual_inputs['labels'],
                                   torch.tensor([[1, 2, 3, -100, -100, -100, -100, -100],
                                                 [1, 2, 3, 4, 5, -100, -100, -100]]))
        loss = CrossEntropyLoss(reduction='sum')
        torch.testing.assert_close(loss(actual_inputs, actual_outputs)['loss'],
                                   loss(expected_inputs, expected_outputs)['loss'])
        # Megatron's embedding hook remains independent of Transformers pooling.
        full_features = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2) + 1
        chunks = []
        for start, end in ((0, 4), (4, 12)):
            pieces = full_features[:, start:end].chunk(4, dim=1)
            chunks.extend((pieces[rank], pieces[3 - rank]))
        local_features = torch.cat(chunks, dim=1)
        pooled = _output_embedding_hook(SimpleNamespace(cp_group=dist.group.WORLD), (),
                                        {'packed_seq_params': full['packed_seq_params']}, local_features)
        expected = torch.nn.functional.normalize(full_features[0, torch.tensor([3, 11])], p=2, dim=1)
        torch.testing.assert_close(pooled, expected)
    finally:
        mpu.get_context_parallel_group = original
        dist.destroy_process_group()


@pytest.mark.slow
def test_megatron_cp_reconstruction_packed_loss_and_embedding_hook():
    import socket
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    mp.spawn(_megatron_processor_cp_worker, args=(port,), nprocs=2, join=True)


def test_processor_filter_preserves_existing_vlm_extension_hook(monkeypatch):
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(InputProcessor, 'VLM_CONCAT_FIELDS',
                        InputProcessor.VLM_CONCAT_FIELDS | {'custom_media_features'})
    value = torch.ones(1, 3, 4)
    output = InputProcessor.to_transformers_dict([{'custom_media_features': value}])[0]
    assert output['custom_media_features'] is value


def test_model_forward_key_ownership_is_preserved_without_sp():
    from types import SimpleNamespace
    from twinkle.processor.fields import resolve_key_specs

    with pytest.raises(ValueError, match='model forward hook'):
        resolve_key_specs(SimpleNamespace(input_key_specs=(DataKeySpec('input_ids', stage='forward'),)))
