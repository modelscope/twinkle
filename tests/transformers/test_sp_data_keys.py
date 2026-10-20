"""Input-rule checks without model training or accelerator allocation."""
import pytest
import torch

from twinkle.model.transformers.strategy.sequence_parallel import DataKeySpec, SequenceParallelStrategy
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
    spec = DataKeySpec('token_weights', split_dim=1, pad_value=-2, need_roll=True)
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
    token_type_spec = DataKeySpec('token_type_ids', split_dim=1, pad_value=-1, stage='forward')
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
    spec = DataKeySpec('token_type_ids', stage='forward')
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
    strategy.register_input_key(DataKeySpec('token_weights', pad_value=-2))
    processor = InputProcessor()
    values = torch.arange(5).unsqueeze(0)
    sample = {'input_ids': torch.ones(1, 5, dtype=torch.long),
              'position_ids': torch.arange(5).unsqueeze(0),
              'labels': torch.ones(1, 5, dtype=torch.long),
              'token_weights': values, 'unregistered': values}
    result = processor([sample], sp_strategy=strategy, input_key_specs=strategy.input_key_specs)
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
    strategy.register_input_key(DataKeySpec('token_features', pad_value=-2, need_gather=True))
    processor = InputProcessor()
    samples = []
    for length in (3, 5):
        samples.append({'input_ids': list(range(1, length + 1)),
                        'position_ids': list(range(length)), 'labels': list(range(1, length + 1)),
                        'token_features': [ [[1] * length, [2] * length] ],
                        'completion_mask': [1] * length})
    result = processor(samples, sp_strategy=strategy, input_key_specs=strategy.input_key_specs)
    expected = torch.tensor([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]])
    torch.testing.assert_close(result['token_features'], expected)
    auxiliary = processor.pop_sp_auxiliary_inputs(result, input_key_specs=strategy.input_key_specs)
    assert set(auxiliary) == {'completion_mask', 'token_features'}
    assert 'token_features' not in result


def test_model_specific_forward_key_is_split_only_in_model_hook(monkeypatch):
    from types import SimpleNamespace
    from twinkle import Platform
    from twinkle.processor import InputProcessor

    monkeypatch.setattr(Platform, 'get_local_device', lambda: torch.device('cpu'))
    strategy = _strategy(1)
    strategy._ctx.tokenizer = SimpleNamespace(pad_token_id=0)
    strategy.register_input_key(DataKeySpec('token_type_ids', stage='forward', owner='model'))

    class Model(torch.nn.Module):
        def forward(self, token_type_ids, **kwargs):
            return token_type_ids

    model = Model()
    strategy._ctx._prepare_forward_hook(model)
    ids = torch.arange(5).unsqueeze(0)
    prepared = InputProcessor()([{'input_ids': ids, 'position_ids': ids,
                                  'token_type_ids': ids}], sp_strategy=strategy,
                                input_key_specs=strategy.input_key_specs)
    assert prepared['token_type_ids'].shape == (1, 5)
    torch.testing.assert_close(model(**prepared), torch.tensor([[3, 4, 0]]))


def test_declared_collection_uses_sequence_axis_and_collects_without_logps(monkeypatch):
    import twinkle.model.transformers.strategy.sequence_parallel as sp

    strategy = _strategy()
    strategy._ctx.extra_kwargs['position_ids'] = torch.arange(5).unsqueeze(0)
    strategy.register_input_key(DataKeySpec('token_features', need_gather=True, pad_value=-2))
    strategy.register_output_key(DataKeySpec('scores', split_dim=2, side='output', stage='output',
                                            need_split=False, need_gather=True, autograd=True))
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

    input_specs = (*BUILTIN_INPUT_KEYS, DataKeySpec('token_features', pad_value=-2, need_gather=True))
    output_specs = (DataKeySpec('scores', side='output', stage='output', need_split=False, pad_value=-3),)
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
    auxiliary = processor.pop_sp_auxiliary_inputs(result, input_key_specs=specs)
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
