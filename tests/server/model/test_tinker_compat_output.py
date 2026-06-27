import torch
from tinker import types

from twinkle.server.common.datum import extract_rl_features_for_loss
from twinkle.server.model.backends.common import TwinkleCompatModelBase


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


def test_tinker_build_output_handles_tensor_rows_from_transformers_backend():
    inputs = [_datum(3), _datum(2)]
    outputs = {'logps': torch.arange(8, dtype=torch.float32).view(2, 4)}

    result = TwinkleCompatModelBase()._tinker_build_output(inputs, outputs)

    assert [item['logprobs'].to_torch().tolist() for item in result] == [[0.0, 1.0, 2.0], [4.0, 5.0]]
    assert [item['elementwise_loss'].to_torch().tolist() for item in result] == [[-0.0, -1.0, -2.0], [-4.0, -5.0]]


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
