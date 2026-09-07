# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from twinkle.data_format import LossOutput
from .cross_entropy import CrossEntropyLoss


class ChannelLoss(CrossEntropyLoss):
    """Cross entropy with per-channel token-loss statistics.

    ``channel`` is sample-level metadata: each batch row belongs to one channel.
    The training objective remains token-level cross entropy; ``channel_loss``
    contains ``[loss_sum, token_count]`` for metric aggregation.
    """

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        labels, per_token, mask = self.get_per_token_loss(inputs, outputs)
        labels, per_token, mask = self._as_batch(labels, per_token, mask)

        channels = self._normalize_channels(inputs.get('channel'), labels.shape[0])
        channel_loss: dict[str | None, torch.Tensor] = {}
        weighted = per_token * mask
        for index, channel in enumerate(channels):
            stats = torch.stack((weighted[index].detach().float().sum(), mask[index].detach().float().sum()))
            if channel in channel_loss:
                channel_loss[channel] = channel_loss[channel] + stats
            else:
                channel_loss[channel] = stats

        result = self._reduce(per_token, mask)
        result['channel_loss'] = channel_loss
        return result

    @staticmethod
    def _as_batch(labels, per_token, mask):
        if labels.ndim == 1:
            return labels.unsqueeze(0), per_token.unsqueeze(0), mask.unsqueeze(0)
        return labels, per_token, mask

    @staticmethod
    def _normalize_channels(channels, batch_size: int):
        if channels is None:
            return [None] * batch_size
        if isinstance(channels, str):
            channels = [channels]
        else:
            channels = list(channels)
        if len(channels) != batch_size:
            raise ValueError(f'channel has {len(channels)} entries, expected one per batch row ({batch_size}).')
        return channels
