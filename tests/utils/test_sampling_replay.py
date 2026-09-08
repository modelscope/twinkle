# Copyright (c) ModelScope Contributors. All rights reserved.
import math

import pytest
import torch

from twinkle.data_format import SamplingMask
from twinkle.utils.torch_utils import replayed_selective_log_softmax


def _sampling_mask(*rows):
    token_ids = [token_id for row in rows for token_id in row]
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return SamplingMask(token_ids=token_ids, offsets=offsets)


def _reference_replayed_logps(logits, labels, loss_mask, sampling_masks, temperature):
    expected = torch.zeros_like(labels, dtype=torch.float32)
    for batch_idx, sampling_mask in enumerate(sampling_masks):
        row_idx = 0
        for seq_idx in loss_mask[batch_idx].nonzero(as_tuple=True)[0].tolist():
            start = sampling_mask.offsets[row_idx]
            end = sampling_mask.offsets[row_idx + 1]
            support = sampling_mask.token_ids[start:end]
            support_logits = logits[batch_idx, seq_idx, support].float() / temperature
            label = int(labels[batch_idx, seq_idx])
            expected[batch_idx, seq_idx] = logits[
                batch_idx, seq_idx, label
            ].float() / temperature - torch.logsumexp(support_logits, dim=0)
            row_idx += 1
    return expected


def test_replayed_logps_match_restricted_softmax_for_ragged_batch():
    logits = torch.tensor(
        [
            [
                [0.2, 1.0, -0.5, 2.0, 0.3],
                [1.1, -0.2, 0.7, 0.1, 2.4],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ],
            [
                [2.0, 0.0, 1.0, -1.0, 0.5],
                [0.4, 1.4, -0.6, 0.2, 0.8],
                [0.9, -0.1, 1.9, 0.3, 0.0],
            ],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[3, 2, -100], [-100, 1, 2]])
    loss_mask = labels != -100
    masks = [
        _sampling_mask([0, 3, 4], [1, 2]),
        _sampling_mask([0, 1, 4], [2]),
    ]

    actual = replayed_selective_log_softmax(
        logits, labels.masked_fill(~loss_mask, 0), loss_mask, masks, 0.7
    )
    expected = _reference_replayed_logps(logits, labels, loss_mask, masks, 0.7)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert torch.equal(actual[~loss_mask], torch.zeros_like(actual[~loss_mask]))
    assert actual[1, 2].item() == 0.0  # A singleton support assigns probability one.


def test_full_vocab_replay_matches_temperature_scaled_log_softmax():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 6)
    labels = torch.tensor([[1, -100, 4], [0, 3, -100]])
    loss_mask = labels != -100
    full_support = list(range(logits.shape[-1]))
    masks = [
        _sampling_mask(full_support, full_support),
        _sampling_mask(full_support, full_support),
    ]

    actual = replayed_selective_log_softmax(
        logits, labels.masked_fill(~loss_mask, 0), loss_mask, masks, temperature=1.3
    )
    expected = (
        torch.log_softmax(logits.float() / 1.3, dim=-1)
        .gather(-1, labels.masked_fill(~loss_mask, 0).unsqueeze(-1))
        .squeeze(-1)
    )
    expected = expected.masked_fill(~loss_mask, 0)

    torch.testing.assert_close(actual, expected)


def test_replay_backward_only_touches_retained_support_logits():
    logits = torch.randn(1, 2, 5, requires_grad=True)
    labels = torch.tensor([[1, 3]])
    mask = _sampling_mask([0, 1, 4], [2, 3])

    replayed_selective_log_softmax(
        logits, labels, torch.ones_like(labels, dtype=torch.bool), [mask], 1.0
    ).sum().backward()

    assert torch.equal(
        logits.grad[0, 0].ne(0), torch.tensor([True, True, False, False, True])
    )
    assert torch.equal(
        logits.grad[0, 1].ne(0), torch.tensor([False, False, True, True, False])
    )


def test_empty_training_batch_returns_zeros():
    logits = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss_mask = torch.zeros_like(labels, dtype=torch.bool)

    result = replayed_selective_log_softmax(
        logits, labels, loss_mask, [SamplingMask([], [0]), SamplingMask([], [0])], 1.0
    )

    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_replay_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        replayed_selective_log_softmax(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            [_sampling_mask([0])],
            temperature,
        )


@pytest.mark.parametrize(
    ("sampling_mask", "message"),
    [
        (None, "missing"),
        (SamplingMask([0], [0, 1]), "1 rows but 2 training tokens"),
        (SamplingMask([0, 5], [0, 1, 2]), "outside vocabulary"),
        (SamplingMask([0, 1], [0, 1, 2]), "absent from sampling mask"),
    ],
)
def test_replay_rejects_malformed_or_incompatible_masks(sampling_mask, message):
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[2, 2]])
    with pytest.raises(ValueError, match=message):
        replayed_selective_log_softmax(
            logits,
            labels,
            torch.ones_like(labels, dtype=torch.bool),
            [sampling_mask],
            1.0,
        )


def test_replay_validates_tensor_and_batch_shapes():
    valid_mask = [_sampling_mask([0])]
    with pytest.raises(ValueError, match="logits must have shape"):
        replayed_selective_log_softmax(
            torch.zeros(1, 2),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            valid_mask,
            1.0,
        )
    with pytest.raises(ValueError, match="labels and loss_mask"):
        replayed_selective_log_softmax(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            valid_mask,
            1.0,
        )
    with pytest.raises(ValueError, match="batch has 0 samples"):
        replayed_selective_log_softmax(
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            [],
            1.0,
        )
