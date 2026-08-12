# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss
from twinkle.loss.grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class PPOValueLoss(Loss):
    """Clipped PPO value-function loss over response tokens."""

    require_logps = False
    require_values = True

    def __init__(self, epsilon: float = 0.2, ignore_index: int = -100, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self._aligner = GRPOLoss(ignore_index=ignore_index)

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_values: Optional[Union['torch.Tensor', List, np.ndarray]] = None,
        returns: Optional[Union['torch.Tensor', List, np.ndarray]] = None,
        **kwargs,
    ) -> LossOutput:
        import torch

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        mask = (labels != self.ignore_index).bool()

        values = outputs.get('values')
        assert values is not None, "outputs must contain 'values'"
        if values.dim() == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        if values.shape != mask.shape:
            raise AssertionError(f'values/mask shape mismatch: values={tuple(values.shape)} mask={tuple(mask.shape)}')
        assert old_values is not None, 'old_values are required for PPO value clipping'
        assert returns is not None, 'returns are required for PPO value loss'

        old_values = self._aligner._pad_and_align_to_batch(old_values, mask, values.device, values.dtype)
        returns = self._aligner._pad_and_align_to_batch(returns, mask, values.device, values.dtype)

        clipped_values = old_values + torch.clamp(values - old_values, -self.epsilon, self.epsilon)
        loss_unclipped = (values - returns).square()
        loss_clipped = (clipped_values - returns).square()
        per_token_loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped)
        mask_f = mask.to(values.dtype)
        loss = (per_token_loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return LossOutput(loss=loss, num_tokens=0)
