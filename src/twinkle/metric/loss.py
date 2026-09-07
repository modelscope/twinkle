# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Union

from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


class LossMetric(Metric):
    """The loss metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
    """

    def __init__(self, device_mesh, process_group, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.total_loss = 0
        self.total_count = 0
        self.grad_norm = 0
        self.num_tokens = 0
        self.channel_loss = {}

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        if 'loss' not in outputs:
            return
        loss = outputs['loss']
        loss_reduction = kwargs.get('loss_reduction', 'mean')
        if loss_reduction == 'sum':
            output_num_tokens = outputs.get('num_tokens')
            if output_num_tokens is not None:
                self.num_tokens += output_num_tokens.item() if hasattr(output_num_tokens, 'item') else output_num_tokens
            else:
                if not isinstance(inputs, list):
                    inputs = [inputs]
                for input in inputs:
                    # Fallback for losses that do not expose an explicit token denominator in outputs.
                    labels = input['labels']
                    self.num_tokens += (labels >= 0).sum().item()
        grad_norm = kwargs.get('grad_norm')
        if grad_norm is not None:
            self.grad_norm = grad_norm

        self.total_loss += loss.item() if hasattr(loss, 'item') else loss
        self.total_count += 1
        for channel, stats in outputs.get('channel_loss', {}).items():
            loss_sum = stats[0].item() if hasattr(stats[0], 'item') else stats[0]
            token_count = stats[1].item() if hasattr(stats[1], 'item') else stats[1]
            current = self.channel_loss.setdefault(channel, [0.0, 0.0])
            current[0] += loss_sum
            current[1] += token_count

    def reset(self):
        self.total_loss = 0
        self.total_count = 0
        self.grad_norm = 0
        self.num_tokens = 0
        self.channel_loss = {}

    def calculate(self):
        local_results = [{
            'loss': self.total_loss,
            'count': self.total_count,
            'grad_norm': self.grad_norm,
            'num_tokens': self.num_tokens,
            'channel_loss': self.channel_loss,
        }]
        all_results = self.gather_results(local_results)

        total_loss = sum(r['loss'] for r in all_results)
        total_count = sum(r['count'] for r in all_results)
        grad_norm = max(r['grad_norm'] for r in all_results)
        num_tokens = sum(r['num_tokens'] for r in all_results)
        channel_loss = {}
        for row in all_results:
            for channel, stats in row['channel_loss'].items():
                current = channel_loss.setdefault(channel, [0.0, 0.0])
                current[0] += stats[0]
                current[1] += stats[1]
        if num_tokens > 0:
            avg_loss = total_loss / num_tokens
        elif total_count > 0:
            avg_loss = total_loss / total_count
        else:
            avg_loss = 0.0
        self.reset()
        results = {}
        if avg_loss is not None:
            results['loss'] = f'{avg_loss:.4f}'
        if grad_norm > 0:
            results['grad_norm'] = f'{grad_norm:.6f}'
        for channel in sorted(channel_loss, key=lambda value: '' if value is None else str(value)):
            loss_sum, token_count = channel_loss[channel]
            if token_count > 0:
                results[f'loss_{channel}'] = f'{loss_sum / token_count:.4f}'
        return results
