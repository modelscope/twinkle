# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward-model (RM) loss: the pairwise Bradley-Terry objective.

A reward model rides a ``num_labels=1`` sequence-classification head (so ``outputs['logits']`` is one
scalar score per sequence, already reduced by the head / processor). Training maximises the margin
between a chosen and a rejected response for the same prompt:

    L_RM = -log(sigmoid(r(chosen) - r(rejected)))

This is distinct from :class:`SeqClsLoss` (which scores each sequence against a fixed label set): RM
has no labels at all, only the *relative* ordering of a chosen/rejected pair, so the batch is laid out
interleaved as ``[chosen_1, rejected_1, chosen_2, rejected_2, ...]`` -- the same layout the DPO family
uses -- and the loss reads the pair off that.
"""
from typing import TYPE_CHECKING

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    import torch


class RewardLoss(Loss):
    """Pairwise Bradley-Terry reward-model loss over interleaved chosen/rejected scores.

    Args:
        center_rewards_coefficient: If > 0, add ``coef * mean((r_chosen + r_rejected) ** 2)`` to keep
            the reward magnitudes centred near zero (as in TRL's ``RewardTrainer``); a reward model is
            identified only up to an additive constant, so without this the scores can drift. None/0
            disables it.
    """

    # Reads outputs['logits'] (the [B, 1] score head), not per-token logps.
    require_logits = True
    require_logps = False

    def __init__(self, center_rewards_coefficient: float = 0.0, **kwargs):
        super().__init__()
        self.center_rewards_coefficient = center_rewards_coefficient or 0.0

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        import torch
        import torch.nn.functional as F

        logits = outputs['logits']
        # The head emits [B, 1]; flatten to [B] so the interleaved split is unambiguous.
        scores = logits.reshape(-1)
        assert scores.numel() % 2 == 0, (
            f'RewardLoss needs an even batch (interleaved chosen/rejected pairs), got {scores.numel()} scores.')
        chosen = scores[0::2]
        rejected = scores[1::2]

        loss = -F.logsigmoid(chosen - rejected).mean()
        if self.center_rewards_coefficient > 0:
            loss = loss + self.center_rewards_coefficient * torch.mean((chosen + rejected)**2)
        return LossOutput(loss=loss, num_tokens=0)
