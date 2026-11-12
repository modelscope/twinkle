# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for InfonceLoss."""
import torch

from twinkle.loss import InfonceLoss
from twinkle.loss.infonce import _parse_multi_negative_sentences


def test_hard_negative_upsampling_never_picks_anchor_or_positive():
    # Two groups laid out as anchor, positive, negatives: [a0, p0, n0] and [a1, p1, n1, n1'].
    sentences = torch.arange(7, dtype=torch.float32).unsqueeze(-1)
    labels = torch.tensor([1, 0, 0, 1, 0, 0, 0])
    groups = _parse_multi_negative_sentences(sentences, labels, hard_negatives=4)
    for group, negatives in zip(groups, ({2.0}, {5.0, 6.0})):
        assert group.shape[0] == 2 + 4
        assert set(group[2:, 0].tolist()) <= negatives


def test_hard_negative_upsampling_keeps_the_loss_of_duplicated_negatives():
    # One negative padded to three copies: the loss equals CE over [positive, neg, neg, neg].
    torch.manual_seed(0)
    anchor, positive, negative = torch.randn(3, 4)
    sentences = torch.stack([anchor, positive, negative])
    loss_fn = InfonceLoss(temperature=1.0, use_batch=False, hard_negatives=3)
    loss = loss_fn({'labels': torch.tensor([1, 0, 0])}, {'embeddings': sentences})['loss']
    logits = torch.stack([anchor @ positive] + [anchor @ negative] * 3)
    expected = -torch.log_softmax(logits, dim=0)[0]
    torch.testing.assert_close(loss, expected)
