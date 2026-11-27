import pytest
import torch
from tinker import types
from unittest.mock import Mock

from twinkle.server.model.tinker_datum import extract_rl_features_for_loss
from twinkle.server.model.backends.common import TwinkleCompatModelBase
from twinkle.server.exceptions import RequestRejectedError


def _datum(seq_len: int, *, ref_logps=None, old_logps=None, advantages=None):
    loss_fn_inputs = {
        'target_tokens': types.TensorData.from_torch(torch.arange(seq_len, dtype=torch.long)),
        'weights': types.TensorData.from_torch(torch.ones(seq_len)),
    }
    if ref_logps is not None:
        loss_fn_inputs['ref_logps'] = types.TensorData.from_torch(torch.tensor(ref_logps, dtype=torch.float32))
    if old_logps is not None:
        loss_fn_inputs['logprobs'] = types.TensorData.from_torch(torch.tensor(old_logps, dtype=torch.float32))
    if advantages is not None:
        loss_fn_inputs['advantages'] = types.TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32))
    return types.Datum(
        model_input=types.ModelInput.from_ints(list(range(seq_len))),
        loss_fn_inputs=loss_fn_inputs,
    )


@pytest.mark.parametrize('loss_fn', ['ppo', 'cispo', 'dro', 'typo'])
def test_tinker_backend_rejects_unsupported_loss_before_changing_adapter(loss_fn):
    model = TwinkleCompatModelBase()
    model.set_loss = Mock()
    kwargs = {'beta': 0.1}

    with pytest.raises(RequestRejectedError, match='Unsupported Tinker loss_fn'):
        model._tinker_setup_loss(loss_fn, [_datum(2, advantages=[-1, -1])], 'adapter', kwargs)

    model.set_loss.assert_not_called()
    assert kwargs == {'beta': 0.1}


@pytest.mark.parametrize('dpo', [False, True])
def test_tinker_backend_preserves_existing_importance_sampling_loss(dpo):
    model = TwinkleCompatModelBase()
    model.set_loss = Mock()
    model._ensure_dpo_metric = Mock()
    inputs = [_datum(2, ref_logps=[-0.1, -0.2]) for _ in range(2)] if dpo else [_datum(2)]

    model._tinker_setup_loss('importance_sampling', inputs, 'adapter', {})

    if dpo:
        model.set_loss.assert_called_once_with(
            'DPOLoss', adapter_name='adapter', beta=0.1, loss_type='sigmoid', sft_weight=0.0)
        model._ensure_dpo_metric.assert_called_once_with('adapter', 0.1)
    else:
        model.set_loss.assert_called_once_with('GRPOLoss', adapter_name='adapter', epsilon=0.2, beta=0.0)


def test_tinker_build_output_handles_tensor_rows_from_transformers_backend():
    inputs = [_datum(3), _datum(2)]
    outputs = {'logps': torch.arange(8, dtype=torch.float32).view(2, 4)}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs)

    assert [item['logprobs'].to_torch().tolist() for item in result] == [[0.0, 1.0, 2.0], [4.0, 5.0]]
    assert [item['elementwise_loss'].to_torch().tolist() for item in result] == [[-0.0, -1.0, -2.0], [-4.0, -5.0]]


def test_tinker_build_output_trims_batch_padded_logps_to_datum_lengths():
    # Regression (G19/C3-SV06): the public tinker forward pass must return one logprob
    # per input token. The transformers backend pads a ragged batch to the max length
    # ([B, T] tensor); the default build path must trim each row back to its datum length
    # so logprobs stays aligned with target_tokens/weights (tinker SDK contract).
    inputs = [_datum(17), _datum(17), _datum(16), _datum(15)]
    outputs = {'logps': torch.arange(4 * 17, dtype=torch.float32).view(4, 17)}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs)

    assert [item['logprobs'].to_torch().shape[0] for item in result] == [17, 17, 16, 15]
    assert [item['elementwise_loss'].to_torch().shape[0] for item in result] == [17, 17, 16, 15]
    # row i keeps the first seq_len_i entries of the padded row
    for i, seq_len in enumerate([17, 17, 16, 15]):
        expected = torch.arange(i * 17, i * 17 + seq_len, dtype=torch.float32)
        torch.testing.assert_close(result[i]['logprobs'].to_torch(), expected)


def test_tinker_build_output_handles_ragged_microbatch_logps_from_megatron_backend():
    inputs = [_datum(66), _datum(65), _datum(64), _datum(63)]
    outputs = {
        'logps': [
            torch.ones(2, 66),
            torch.full((2, 65), 2.0),
        ]
    }

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs)

    assert [item['logprobs'].to_torch().shape[0] for item in result] == [66, 65, 64, 63]
    torch.testing.assert_close(result[0]['logprobs'].to_torch(), torch.ones(66))
    torch.testing.assert_close(result[1]['logprobs'].to_torch(), torch.ones(65))
    torch.testing.assert_close(result[2]['logprobs'].to_torch(), torch.full((64, ), 2.0))
    torch.testing.assert_close(result[3]['logprobs'].to_torch(), torch.full((63, ), 2.0))


def test_tinker_build_output_ignores_empty_pipeline_stage_outputs():
    inputs = [_datum(3), _datum(2)]

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, {'logps': []})

    assert result == []


def test_tinker_build_output_handles_ragged_microbatch_logits_fallback():
    inputs = [_datum(2), _datum(1), _datum(1)]
    logits = [
        torch.tensor([
            [[10.0, 0.0], [0.0, 10.0]],
            [[0.0, 10.0], [10.0, 0.0]],
        ]),
        torch.tensor([[[10.0, 0.0]]]),
    ]

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, {'logits': logits})

    assert len(result) == 3
    assert [item['logprobs'].to_torch().shape[0] for item in result] == [2, 1, 1]


def test_tinker_build_output_keeps_full_ragged_logps_for_dpo_reference_forward():
    inputs = [_datum(3), _datum(2)]
    outputs = {
        'logps': [
            torch.arange(5, dtype=torch.float32).view(1, 5),
            torch.arange(4, dtype=torch.float32).view(1, 4),
        ]
    }

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs, return_full_logprobs=True)

    assert [item['logprobs'].to_torch().tolist() for item in result] == [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0],
    ]


def test_tinker_build_output_splits_single_packed_logps_row():
    inputs = [_datum(3), _datum(2)]
    outputs = {'logps': torch.arange(5, dtype=torch.float32).view(1, 5)}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs, return_full_logprobs=True)

    assert [item['logprobs'].to_torch().tolist() for item in result] == [[0.0, 1.0, 2.0], [3.0, 4.0]]


def test_tinker_build_output_splits_batched_row_from_megatron_reference_forward():
    inputs = [_datum(4), _datum(4)]
    outputs = {'logps': torch.arange(8, dtype=torch.float32).view(1, 2, 4)}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs, return_full_logprobs=True)

    assert [item['logprobs'].to_torch().tolist() for item in result] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_tinker_build_output_does_not_treat_packed_logits_as_batch_major():
    inputs = [_datum(1), _datum(1)]
    outputs = {'logits': torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs)

    assert len(result) == 2
    assert [item['logprobs'].to_torch().shape for item in result] == [torch.Size([1]), torch.Size([1])]


def test_tinker_build_output_squeezes_singleton_reference_logps():
    inputs = [_datum(4)]
    outputs = {'logps': [torch.arange(4, dtype=torch.float32).view(1, 4)]}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs, return_full_logprobs=True)

    assert result[0]['logprobs'].to_torch().shape == torch.Size([4])
    assert result[0]['logprobs'].to_torch().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_extract_rl_features_pads_ragged_dpo_ref_logps():
    result = extract_rl_features_for_loss([
        _datum(3, ref_logps=[1.0, 2.0, 3.0]),
        _datum(2, ref_logps=[4.0, 5.0]),
    ])

    torch.testing.assert_close(result['ref_outputs']['logps'], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]]))


def test_extract_rl_features_keeps_grpo_logps_and_advantages_ragged():
    result = extract_rl_features_for_loss([
        _datum(3, old_logps=[1.0, 2.0, 3.0], advantages=[0.5, 0.5, 0.5]),
        _datum(2, old_logps=[4.0, 5.0], advantages=[-0.5, -0.5]),
    ])

    assert result['old_logps'] == [[1.0, 2.0, 3.0], [4.0, 5.0]]
    assert result['advantages'] == [[0.5, 0.5, 0.5], [-0.5, -0.5]]


def test_normalize_ref_outputs_flattens_and_pads_serialized_logps():
    kwargs = {
        'ref_outputs': {
            'logps': [
                [[1.0, 2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0]],
            ]
        }
    }

    TwinkleCompatModelBase._normalize_ref_outputs(kwargs)

    torch.testing.assert_close(
        kwargs['ref_outputs']['logps'],
        torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 0.0],
            [6.0, 7.0, 0.0],
        ]),
    )


def test_normalize_ref_outputs_keeps_tensor_unchanged():
    logps = torch.tensor([[1.0, 2.0]])
    kwargs = {'ref_outputs': {'logps': logps}}

    TwinkleCompatModelBase._normalize_ref_outputs(kwargs)

    assert kwargs['ref_outputs']['logps'] is logps
