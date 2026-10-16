# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Union

from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


class PPOValueMetric(Metric):
    """Aggregate PPO critic statistics over valid response tokens."""

    def __init__(self, device_mesh=None, process_group=None, epsilon: float = 0.2, ignore_index: int = -100, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.records = []

    def accumulate(self,
                   inputs: Union[InputFeature, List[InputFeature]],
                   outputs: ModelOutput,
                   *,
                   old_values=None,
                   returns=None,
                   advantages=None,
                   **kwargs):
        import torch

        from twinkle.utils.transformers_utils import align_logps_to_mask

        if outputs is None or outputs.get('values') is None or old_values is None or returns is None:
            return
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        values_list = outputs['values'] if isinstance(outputs['values'], list) else [outputs['values']]
        cursor = 0
        for mb_input, values in zip(inputs_list, values_list):
            labels = torch.as_tensor(mb_input['labels'], device=values.device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if values.dim() == 1:
                values = values.unsqueeze(0)
            mask = labels != self.ignore_index
            batch_size = labels.shape[0]
            old = align_logps_to_mask(old_values[cursor:cursor + batch_size], mask, values.dtype)
            target = align_logps_to_mask(returns[cursor:cursor + batch_size], mask, values.dtype)
            adv = (
                align_logps_to_mask(advantages[cursor:cursor
                                               + batch_size], mask, values.dtype) if advantages is not None else None)
            cursor += batch_size
            if old is None or target is None or not mask.any():
                continue
            clipped = old + (values - old).clamp(-self.epsilon, self.epsilon)
            self.records.append({
                'values': values[mask].detach().float().cpu().tolist(),
                'returns': target[mask].detach().float().cpu().tolist(),
                'advantages': adv[mask].detach().float().cpu().tolist() if adv is not None else [],
                'clipped': ((values - old).abs() > self.epsilon)[mask].detach().float().cpu().tolist(),
                'clipped_values': clipped[mask].detach().float().cpu().tolist(),
            })

    def calculate(self) -> Dict[str, Any]:
        import torch

        records = self.gather_results(self.records)
        self.reset()
        if not records:
            return {}
        values = torch.tensor([v for record in records for v in record['values']])
        returns = torch.tensor([v for record in records for v in record['returns']])
        clipped = torch.tensor([v for record in records for v in record['clipped']])
        advantages = torch.tensor([v for record in records for v in record['advantages']])
        return_var = returns.var(unbiased=False)
        explained_variance = 1.0 - (returns - values).var(unbiased=False) / return_var.clamp(min=1e-8)
        result = {
            'train/value_mean': values.mean().item(),
            'train/return_mean': returns.mean().item(),
            'train/value_clip_ratio': clipped.mean().item(),
            'train/explained_variance': explained_variance.item(),
        }
        if advantages.numel():
            result['train/advantage_mean'] = advantages.mean().item()
            result['train/advantage_std'] = advantages.std(unbiased=False).item()
        return result
